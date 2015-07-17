#include <iostream>
#include <fstream>
#include <iomanip>
#include "image.h"

#define BUF_SIZE 256

using namespace std;

class errorPNM { };

struct Color
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

void readPNM(ifstream &file, char* buf);
image<unsigned char>* loadPGM(const char* name);
void savePPM(image<Color>* im, const char* name);
Color randomColor();

__global__ void evolveContour(unsigned char* intensityDev, unsigned char* labelsDev, signed char* speedDev, signed char* phiDev, int HEIGHT, int WIDTH, int* targetLabels, int kernelID, int numLabels, int* lowerIntensityBounds, int* upperIntensityBounds);

__global__ void initSpeedPhi(unsigned char* intensity, unsigned char* labels, signed char* speed, signed char* phi, int HEIGHT, int WIDTH, int targetLabel, int lowerIntensityBound, int upperIntensityBound);

__global__ void switchIn(signed char* speed, signed char* phi, int HEIGHT, int WIDTH);
__global__ void switchOut(signed char* speed, signed char* phi, int HEIGHT, int WIDTH);

__global__ void checkStopCondition(signed char* speed, signed char* phi, int parentThreadID, int HEIGHT, int WIDTH);
__device__ volatile int stopCondition[1024];

void usage()
{
	cout<<"Usage: ./lss <Input intensities path> <Input labels path> <Input params path> <GOLD output path> <#repetitions (HyperQ)>" << endl;
}

int main(int argc, char* argv[])
{
	// Parse command line arguments
	if(argc < 6)
    {
		usage();
		exit(0);
    }
	char* imageFile = argv[1];
	char* labelFile = argv[2];
	char* paramFile = argv[3];
	char* outputFile = argv[4];
	int numRepetitions = atoi(argv[5]);

        // Initialize timers, start the runtime timer
	cudaEvent_t startTime1, startTime2, stopTime1, stopTime2;
	cudaEventCreate(&startTime1);
	cudaEventCreate(&startTime2);
	cudaEventCreate(&stopTime1);
	cudaEventCreate(&stopTime2);
	float elapsedTime1, elapsedTime2;
	cudaEventRecord(startTime1, 0);


        // Load image, send to GPU
	image<unsigned char>* input = loadPGM(imageFile);
	const int HEIGHT = input->height();
	const int WIDTH = input->width();
	const int SIZE = HEIGHT*WIDTH*sizeof(char);

	unsigned char* intensity = new unsigned char[numRepetitions*HEIGHT*WIDTH];
	for(int i=0; i<numRepetitions; i++)
		memcpy(&intensity[i*HEIGHT*WIDTH], input->data, SIZE);

	unsigned char* intensityDev = NULL;
	cudaMalloc((void**)&intensityDev, numRepetitions*SIZE);
	cudaMemcpyAsync(intensityDev, intensity, numRepetitions*SIZE, cudaMemcpyHostToDevice);


        // Load connected component labels, send to GPU
	input = loadPGM(labelFile);

	unsigned char* labels = new unsigned char[numRepetitions*HEIGHT*WIDTH];
	for(int i=0; i<numRepetitions; i++)
		memcpy(&labels[i*HEIGHT*WIDTH], input->data, SIZE);

	unsigned char* labelsDev = NULL;
	cudaMalloc((void **)&labelsDev, numRepetitions*SIZE);
	cudaMemcpyAsync(labelsDev, labels, numRepetitions*SIZE, cudaMemcpyHostToDevice);


	// Load parameters, send to GPU
	ifstream paramStream;
	paramStream.open(paramFile);

	if(paramStream.is_open() != true)
	{
		cerr << "Could not open '" << paramFile << "'." << endl;
		exit(1);
	}

	int targetLabels[1024];
	int lowerIntensityBounds[1024];
	int upperIntensityBounds[1024];

	int numLabels = 0;
	while(paramStream.eof() == false)
	{
		char line[16];
		paramStream.getline(line, 16);
		
		if(paramStream.eof() == true)
			break;

		if(numLabels % 3 == 0)
			targetLabels[numLabels/3] = strtol(line, NULL, 10);
		else if(numLabels % 3 == 1)
			lowerIntensityBounds[numLabels/3] = strtol(line, NULL, 10);
		else
			upperIntensityBounds[numLabels/3] = strtol(line, NULL, 10);
		
		numLabels++;
	}
	
	if(numLabels % 3 == 0)
		numLabels /= 3;
	else
	{
		cerr << "Number of lines in " << paramFile << " is not divisible by 3. Try '" << argv[0]
			<< " --help' for additional information." << endl;
		exit(1);
	}
	paramStream.close();

	int* targetLabelsDev = NULL;
        cudaMalloc((void**)&targetLabelsDev, numLabels*sizeof(int));
        cudaMemcpyAsync(targetLabelsDev, targetLabels, numLabels*sizeof(int), cudaMemcpyHostToDevice);

        int* lowerIntensityBoundsDev = NULL;
        cudaMalloc((void**)&lowerIntensityBoundsDev, numLabels*sizeof(int));
        cudaMemcpyAsync(lowerIntensityBoundsDev, lowerIntensityBounds, numLabels*sizeof(int), cudaMemcpyHostToDevice);

        int* upperIntensityBoundsDev = NULL;
        cudaMalloc((void**)&upperIntensityBoundsDev, numLabels*sizeof(int));
        cudaMemcpyAsync(upperIntensityBoundsDev, upperIntensityBounds, numLabels*sizeof(int), cudaMemcpyHostToDevice);


        // Allocate arrays for speed and phi in GPU memory
	signed char* speedDev = NULL;
	signed char* phiDev = NULL;
	cudaMalloc((void**)&speedDev, numRepetitions*numLabels*SIZE);
	cudaMalloc((void**)&phiDev, numRepetitions*numLabels*SIZE);

	cudaDeviceSynchronize();


	// Start the segmentation timer
	cudaEventRecord(startTime2, 0);
	

	// Launch kernel to begin image segmenation
	for(int i=0; i<numRepetitions; i++)
	{
		evolveContour<<<1, numLabels>>>(intensityDev, labelsDev, speedDev, phiDev, HEIGHT, WIDTH, targetLabelsDev, i,
						numLabels, lowerIntensityBoundsDev, upperIntensityBoundsDev);
	}
	cudaDeviceSynchronize();


	// Stop the segmentation timer
	cudaEventRecord(stopTime2, 0);


	// Retrieve results from the GPU
	signed char* phi = new signed char[numRepetitions*numLabels*HEIGHT*WIDTH];
	cudaMemcpy(phi, phiDev, numRepetitions*numLabels*SIZE, cudaMemcpyDeviceToHost);


	// Stop the runtime timer
	cudaEventRecord(stopTime1, 0);


	// Caio: Output: DEV
	FILE *fout;
	fout = fopen(outputFile, "wb");
	if (!fout)
	{
		printf("Could not open output file. %s\n", outputFile);
		exit(0);
	}
	fwrite(phi, numRepetitions*numLabels*SIZE, 1, fout);
	fclose(fout);
	printf("GOLD written to file.\n");


        // Stop runtime timer and print times
	cudaEventElapsedTime(&elapsedTime1, startTime1, stopTime1);
	cudaEventElapsedTime(&elapsedTime2, startTime2, stopTime2);
	cout << "Computation time: " << setprecision(6) << elapsedTime2 << " ms"<< endl;
	cout << "Total time: " << setprecision(6) << elapsedTime1 << " ms"<< endl;
	

	// Free resources and end the program
	cudaEventDestroy(startTime1);
	cudaEventDestroy(stopTime1);
	cudaEventDestroy(startTime2);
	cudaEventDestroy(stopTime2);

	cudaFree(intensityDev);
	cudaFree(labelsDev);
	cudaFree(speedDev);
	cudaFree(phiDev);
	cudaFree(targetLabelsDev);
	cudaFree(lowerIntensityBoundsDev);
	cudaFree(upperIntensityBoundsDev);

        return 0;
}


image<unsigned char>* loadPGM(const char* name)
{
	char buf[BUF_SIZE];

	// Read header
	ifstream file(name, ios::in | ios::binary);
	readPNM(file, buf);
	if(strncmp(buf, "P5", 2))
	{
		cerr << "Unable to open '" << name << "'." << endl;
		throw errorPNM();
	}

	readPNM(file, buf);
	int width = atoi(buf);
	readPNM(file, buf);
	int height = atoi(buf);

	readPNM(file, buf);
	if(atoi(buf) > UCHAR_MAX)
	{
		cerr << "Unable to open '" << name << "'." << endl;
		throw errorPNM();
	}

	// Read data
	image<unsigned char>* im = new image<unsigned char>(width, height);
	file.read((char*)imPtr(im, 0, 0), width*height*sizeof(unsigned char));

	return im;
}


void readPNM(ifstream &file, char* buf)
{
	char doc[BUF_SIZE];
	char c;

	file >> c;
	while (c == '#')
	{
		file.getline(doc, BUF_SIZE);
		file >> c;
	}
	file.putback(c);

	file.width(BUF_SIZE);
	file >> buf;
	file.ignore();
}


void savePPM(image<Color>* im, const char* name)
{
	int width = im->width();
	int height = im->height();
	ofstream file(name, ios::out | ios::binary);

	file << "P6\n" << width << " " << height << "\n" << UCHAR_MAX << "\n";
	file.write((char*)imPtr(im, 0, 0), width*height*sizeof(Color));
}


Color randomColor()
{
	Color c;
	c.r = (unsigned char) rand();
	c.g = (unsigned char) rand();
	c.b = (unsigned char) rand();

	return c;
}


__global__ void evolveContour(unsigned char* intensity, unsigned char* labels, signed char* speed, signed char* phi, int HEIGHT, int WIDTH, int* targetLabels, int kernelID, int numLabels, int* lowerIntensityBounds, int* upperIntensityBounds)
{
        int tid = threadIdx.x;

        intensity = &intensity[kernelID*HEIGHT*WIDTH];
        labels = &labels[kernelID*HEIGHT*WIDTH];
        speed = &speed[(kernelID*numLabels+tid)*HEIGHT*WIDTH];
        phi = &phi[(kernelID*numLabels+tid)*HEIGHT*WIDTH];

        dim3 dimGrid(WIDTH/30+1, HEIGHT/30+1);
        dim3 dimBlock(32, 32);
        initSpeedPhi<<<dimGrid, dimBlock>>>(intensity, labels, speed, phi, HEIGHT, WIDTH, targetLabels[tid], lowerIntensityBounds[tid], upperIntensityBounds[tid]);

        int numIterations = 0;
        stopCondition[tid] = 1;
        while(stopCondition[tid])
        {
                stopCondition[tid] = 0;
                numIterations++;
                dimGrid.x = WIDTH/30+1;
                dimGrid.y = HEIGHT/30+1;
 
		// Outward evolution
                switchIn<<<dimGrid, dimBlock>>>(speed, phi, HEIGHT, WIDTH);

                // Inward evolution
                switchOut<<<dimGrid, dimBlock>>>(speed, phi, HEIGHT, WIDTH);

                // Check stopping condition on every third iteration
                if(numIterations % 3 == 0)
                {
                        dimGrid.x = WIDTH/32+1;
                        dimGrid.y = HEIGHT/32+1;
                        checkStopCondition<<<dimGrid, dimBlock>>>(speed, phi, tid, HEIGHT, WIDTH);
                        cudaDeviceSynchronize();
                }
		else
			stopCondition[tid] = 1;

                if(stopCondition[tid] == 0)
                	printf("Target label %d (intensities: %d-%d) converged in %d iterations.\n", targetLabels[tid], lowerIntensityBounds[tid], upperIntensityBounds[tid], numIterations);
	}
}


__global__ void initSpeedPhi(unsigned char* intensity, unsigned char* labels, signed char* speed, signed char* phi, int HEIGHT, int WIDTH, int targetLabel, int lowerIntensityBound, int upperIntensityBound)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int xPos = 30*bx + tx;
	int yPos = 30*by + ty;

	int intensityReg;
	int speedReg;
	int phiReg;
	__shared__ int labelsTile[32][32];

	// Load data into shared memory and registers
	if(xPos < WIDTH && yPos < HEIGHT)
	{
		labelsTile[ty][tx] = labels[yPos*WIDTH+xPos];
		intensityReg = intensity[yPos*WIDTH+xPos];
	}

	// Initialization
	if(tx > 0 && tx < 31 && ty > 0 && ty < 31 && xPos < WIDTH-1 && yPos < HEIGHT-1)
	{
		// Phi
		if(labelsTile[ty][tx] != targetLabel)
		{
			if(labelsTile[ty][tx-1] != targetLabel && labelsTile[ty][tx+1] != targetLabel && labelsTile[ty-1][tx] != targetLabel && labelsTile[ty+1][tx] != targetLabel)
				phiReg = 3;
			else
				phiReg = 1;
		}
		else
		{
			if(labelsTile[ty][tx-1] != targetLabel || labelsTile[ty][tx+1] != targetLabel || labelsTile[ty-1][tx] != targetLabel || labelsTile[ty+1][tx] != targetLabel)
				phiReg = -1;
			else
				phiReg = -3;
		}

		// Speed
		if(intensityReg >= lowerIntensityBound && intensityReg <= upperIntensityBound)
			speedReg = 1;
		else
			speedReg = -1;

		// Load data back into global memory
		speed[yPos*WIDTH+xPos] = speedReg;
		phi[yPos*WIDTH+xPos] = phiReg;
	}
}


__global__ void switchIn(signed char* speed, signed char* phi, int HEIGHT, int WIDTH)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int xPos = 30*bx + tx;
	int yPos = 30*by + ty;

	int speedReg;
	__shared__ int phiTile[32][32];

	// Load data into shared memory and registers
	if(xPos < WIDTH && yPos < HEIGHT)
	{
		speedReg = speed[yPos*WIDTH+xPos];
		phiTile[ty][tx] = phi[yPos*WIDTH+xPos];
	}

	if(xPos > 0 && xPos < WIDTH-1 && yPos > 0 && yPos < HEIGHT-1)
	{
		// Delete points from Lout and add them to Lin
		if(phiTile[ty][tx] == 1 && speedReg > 0)
			phiTile[ty][tx] = -1;

		if(tx > 0 && tx < 31 && ty > 0 && ty < 31)
		{
			// Update neighborhood
			if(phiTile[ty][tx] == 3)
			{
				if(phiTile[ty][tx-1] == -1 || phiTile[ty][tx+1] == -1 || phiTile[ty-1][tx] == -1 || phiTile[ty+1][tx] == -1)
					phiTile[ty][tx] = 1;
			}

			// Eliminate redundant points in Lin
			if(phiTile[ty][tx] == -1)
			{
				if(phiTile[ty][tx-1] < 0 && phiTile[ty][tx+1] < 0 && phiTile[ty-1][tx] < 0 && phiTile[ty+1][tx] < 0)
					phiTile[ty][tx] = -3;
			}

			// Load data back into global memory
			phi[yPos*WIDTH+xPos] = phiTile[ty][tx];
		}
	}
}


__global__ void switchOut(signed char* speed, signed char* phi, int HEIGHT, int WIDTH)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int xPos = 30*bx + tx;
	int yPos = 30*by + ty;

	int speedReg;
	__shared__ int phiTile[32][32];

	// Load data into shared memory and registers
	if(xPos < WIDTH && yPos < HEIGHT)
	{
		speedReg = speed[yPos*WIDTH+xPos];
		phiTile[ty][tx] = phi[yPos*WIDTH+xPos];
	}

	if(xPos > 0 && xPos < WIDTH-1 && yPos > 0 && yPos < HEIGHT-1)
	{
		// Delete points from Lin and add them to Lout
		if(phiTile[ty][tx] == -1 && speedReg < 0)
			phiTile[ty][tx] = 1;

		if(tx > 0 && tx < 31 && ty > 0 && ty < 31)
		{
			// Update neighborhood
			if(phiTile[ty][tx] == -3)
			{
				if(phiTile[ty][tx-1] == 1 || phiTile[ty][tx+1] == 1 || phiTile[ty-1][tx] == 1 || phiTile[ty+1][tx] == 1)
					phiTile[ty][tx] = -1;
			}

			// Eliminate redundant points
			if(phiTile[ty][tx] == 1)
			{
				if(phiTile[ty][tx-1] > 0 && phiTile[ty][tx+1] > 0 && phiTile[ty-1][tx] > 0 && phiTile[ty+1][tx] > 0)
					phiTile[ty][tx] = 3;
			}

			// Load data back into global memory
			phi[yPos*WIDTH+xPos] = phiTile[ty][tx];
		}
	}

}


__global__ void checkStopCondition(signed char* speed, signed char* phi, int parentThreadID, int HEIGHT, int WIDTH)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int xPos = 32*bx + tx;
	int yPos = 32*by + ty;

	int speedReg;
	int phiReg;

	// Load data into shared memory and registers
	if(xPos < WIDTH && yPos < HEIGHT)
	{
		speedReg = speed[yPos*WIDTH+xPos];
		phiReg = phi[yPos*WIDTH+xPos];
	}

	// Falsify stop condition if criteria are not met
	if(phiReg == 1 && speedReg > 0)
		stopCondition[parentThreadID]=0;
	else if(phiReg == -1 && speedReg < 0)
		stopCondition[parentThreadID]=1;
}
