//OpenCL deprecated APIs and exception handling 
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

//Processing modes
#define SERIAL "SERIAL"
#define PARALLEL "PARALLEL"

//Include OpenCL and other necessary libraries
#include <C:\Users\Marcus\Desktop\CAT3024N_Assignment\Tutorial 1\CL\cl2.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <algorithm>
#include <sstream>
#include <iomanip>

//Include for histogram visualization
#include "C:\MinGW\include\GL\glut.h"

// Include custom utility, weather data, and serial statistics header
#include "Utils.h"
#include "Weather.h"
#include "SerialStatistics.h"

//Define MyType
typedef float myType;

//Declarations of functions
void Menu();
// serial functions
void Serial(std::vector<myType>& Values, bool summary);
void selectionSort(std::vector<myType>& Values);
void SerialSplitStates(std::vector<float>& temp, std::vector<string>& stateName);
void SerialSplitMonths(std::vector<float>& temp, std::vector<int>& month);
// parallel functions
void Parallel(std::vector<float>& Values, cl::Context context, cl::CommandQueue queue, cl::Program program, cl::Event& prof_event, bool summary);
void ParallelSplitStates(std::vector<float>& temp, cl::Context context, cl::CommandQueue queue, cl::Program program, cl::Event& prof_event, std::vector<string>& stateName);
void ParallelSplitMonths(std::vector<float>& temp, cl::Context context, cl::CommandQueue queue, cl::Program program, cl::Event& prof_event, std::vector<int>& month);
myType SumVec(std::vector<myType>& temp, cl::Context context, cl::CommandQueue queue, cl::Program program, cl::Event& prof_event);
myType STDVec(std::vector<myType>& temp, myType Mean, cl::Context context, cl::CommandQueue queue, cl::Program program, cl::Event& prof_event);
void Sort(std::vector<myType>& temp, cl::Context context, cl::CommandQueue queue, cl::Program program, cl::Event& prof_event);
int AddPadding(std::vector<myType>& temp, size_t LocalSize, float PadVal);
void KernalExec(cl::Kernel kernel, std::vector<myType>& temp, size_t Local_Size, cl::Context context, cl::CommandQueue queue, bool Two, bool Three, bool Four, float FThree, int IFour, cl::Event& prof_event, std::string Name);
float KernalExecRet(cl::Kernel kernel, std::vector<myType>& temp, size_t Local_Size, cl::Context context, cl::CommandQueue queue, bool Two, bool Three, bool Four, float FThree, int IFour, cl::Event& prof_event, std::string Name);
// histogram functions
std::vector<float> updateHistogram(Weather data);
void serialHistogram(std::vector<float>& temperature, float minimum, float maximum);
void parallelHistogram(std::vector<float>& temperature, cl::Context context, cl::CommandQueue queue, cl::Program program, cl::Event& prof_event, float minimum, float maximum);
void initHistogram(void);
void renderHistogram(void);
void reshapeWindow(int w, int h);
void drawLine(GLint x1, GLint y1, GLint x2, GLint y2);
void drawRect(GLint x1, GLint y1, GLint x2, GLint y2);
void drawText(const char* text, GLint length, GLint x, GLint y, void* font);


// Global control variables for data paths, states and months
const string dataset_path = "Brazil_air_temp.txt";
const string states[5] = { "BERNADO", "GUARULHOS", "SANTOS", "TARAUACA", "CACOAL" };
const string months[12] = { "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December" };

//Global flags and settings for program execution and visualization
bool showMenu = true; // if false then need to control variables underneath to show the result
bool showHist = true; // True: display histogram; False: display statistics only
bool exitProgram = false; // Flag to indicate program termination
string method = SERIAL; // Choose between SERIAL or PARALLEL processing
int histogram_mode = 0; // Histogram modes: [0: overall; 1: state; 2: month]
int histogram_state = 3; // States index for histogram: [1: BERNADO; 2: GUARULHOS; 3: SANTOS; 4: TARAUACA; 5: CACOAL] 
int histogram_month = 12; // Months index for histogram [1: January, 2: February, ..., 12: December]
int histogram_bin_no = 5; // Number of bins in the histogram
const int histogram_wheight = 480; // Height of the histogram window
const int histogram_wwidth = 720; // Width of the histogram window

// Global variables for histogram data
std::vector<float> upperLimits; // upper limit for each bins
std::vector<int> frequencies; // store frequency of each bins
// Global object statistical calculations in serial processing 
SerialStatistics SStats = SerialStatistics();


//Print help function
void print_help()
{
	//Display help message
	std::cout << "Application usage:" << std::endl;
	std::cout << "  -p : select platform " << std::endl;
	std::cout << "  -d : select device" << std::endl;
	std::cout << "  -l : list all platforms and devices" << std::endl;
	std::cout << "  -h : print this message" << std::endl;
}

//Main method
int main(int argc, char** argv)
{
	//Handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	//Loop over arguments and perform functions
	for (int i = 1; i < argc; i++)
	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}
	//Try to get all the relecant information ready
	try
	{
		//Get the context
		cl::Context context = GetContext(platform_id, device_id);
		//Display what platform and device the code is running on
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
		//Get the queue
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);
		//Setup the sources
		cl::Program::Sources sources;
		//Link to the kernals.cl document
		AddSources(sources, "my_kernels.cl");
		//Define the program with the context and sources
		cl::Program program(context, sources);
		//Setup prof Event
		cl::Event prof_event;

		//Try to build the program
		try
		{
			program.build();
		}
		catch (const cl::Error& err)
		{
			//Else display error messages
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Load Weather Data
		Weather Data = Weather();
		// if successfully load the dataset
		if (Data.Load(dataset_path)) {
			//Get the temperature data (degree celcius) from the weather data
			std::vector<float>& temps = Data.GetTemp();
			//Get the state name data from the weather data
			std::vector<string>& stateNames = Data.GetName();
			std::vector<int>& months = Data.GetMonth();

			//Load menu
			if (showMenu) {
				Menu();
				system("cls"); // after user select an option, clear screen for better view
			}

			// terminate the program
			if (exitProgram) return 0;

			std::cout << method <<std::endl << "loading... Please wait" << std::endl << std::endl;

			if (showHist) { // show histogram				
				float startTime = clock();
				std::vector<float> partTemp = updateHistogram(Data);

				if (method == SERIAL) {
					selectionSort(partTemp);
					serialHistogram(partTemp, partTemp[0], partTemp[partTemp.size() - 1]);
				}
				else {
					Sort(partTemp, context, queue, program, prof_event);
					parallelHistogram(partTemp, context, queue, program, prof_event, partTemp[0], partTemp[partTemp.size() - 1]);
				}

				float endTime = clock();
				std::cout << "TOTAL TIME: \t" << (endTime - startTime) << " ms" << std::endl;

				glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);				// single frame buffer with RGB color
				glutInitWindowPosition(100, 100);							// window position on the screen
				glutInitWindowSize(histogram_wwidth, histogram_wheight);	// OpenGL Window size
				glutCreateWindow("Histogram");								// create histogram window
				glutDisplayFunc(renderHistogram);							// load render function
				glutReshapeFunc(reshapeWindow);								// load reshape function
				initHistogram();											// Initialize the glut window
				glutMainLoop();												// loop frame until stop
			}
			else { // show statistic 
				if (method == SERIAL) {
					Serial(temps, true);
					SerialSplitStates(temps, stateNames);
					SerialSplitMonths(temps, months);
				}
				else {
					//Perform all tasks in parallel
					Parallel(temps, context, queue, program, prof_event, true);
					ParallelSplitStates(temps, context, queue, program, prof_event, stateNames);
					ParallelSplitMonths(temps, context, queue, program, prof_event, months);
				}
			}
		}
	}
	catch (cl::Error err)
	{
		//Else display error message
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}

	system("pause");
	return 0;
}

void Menu() {
	int option,options = 0, sub_option;

	std::cout << "Select your desired operation:" << std::endl;
	std::cout << "------------------------------" << std::endl;
	std::cout << "(1) For Parallel Procesing" << std::endl;
	std::cout << "(2) For Serial Procesing" << std::endl;
	std::cout << "SYSTEM CONTROL:" << std::endl;
	std::cout << "(3) Exit Program" << std::endl;\
	std::cout << "Please enter your choice: " << std::endl;
	std::cin.clear(); // clear the error flag
	std::cin >> option; // get user input
	std::cout << "------------------------------" << std::endl;

	while (option < 1 || option > 3 || std::cin.fail()) {
		std::cout << "Invalid input! Try again!" << std::endl;
		std::cin.clear(); // clear the error flag
		std::cin.ignore(numeric_limits<streamsize>::max(), '\n');
		std::cout << "Option: ";
		std::cin >> option; // get input
		std::cout << "------------------------------" << std::endl;
	}
	if (option == 1)
	{
		std::cout << "Parallel Processing Option:" << std::endl;
		std::cout << "(1) Compute Summary Statistics (Parallel)" << std::endl;
		std::cout << "(2) Generate Summary Histogram (Parallel)" << std::endl;
		std::cout << "(3) Create State-wise Histogram (Parallel)" << std::endl;
		std::cout << "(4) Create Month-wise Histogram (Parallel)" << std::endl;
		std::cout << "Please enter your choice: " << std::endl;
		std::cin >> options;
	}
	else if(option == 2)
	{
		std::cout << "Serial Processsing Option:" << std::endl;
		std::cout << "(5) Compute Summary Statistics (Serial)" << std::endl;
		std::cout << "(6) Generate Summary Histogram (Serial)" << std::endl;
		std::cout << "(7) Create State-wise Histogram (Serial)" << std::endl;
		std::cout << "(8) Create Month-wise Histogram (Serial)" << std::endl;
		std::cout << "Please enter your choice: " << std::endl;
		std::cin >> options;
	}
	else if (option == 3)
	{
		exitProgram = true;
	}

	// Start to tune the global variable based on user choice
	if (options >= 5 && options <= 8) { // serial method
		method = SERIAL;
		showHist = (options != 1) ? true : false; // if not option 1, show histogram
	}
	else if (options >= 1 && options <= 4) { // parallel method
		method = PARALLEL;
		showHist = (options != 5) ? true : false; // if not option 5, show histogram
	}

	// show summary histogram
	if (options == 2 || options == 6) {
		histogram_mode = 0;
	}

	// show state histogram
	if (options == 3 || options == 7) {
		histogram_mode = 1;
		std::cout << "Choose a state: " << std::endl;
		for (int i = 1; i < 6; i++) {
			std::cout << i << ". " << states[i - 1] << std::endl;
		}
		std::cout << "------------------------------" << std::endl;
		std::cout << "Option: ";
		std::cin.clear();
		std::cin >> sub_option;
		std::cout << "------------------------------" << std::endl;

		// if invalid input
		while (sub_option < 1 || sub_option > 5 || std::cin.fail()) {
			std::cout << "Invalid input! Try again!" << std::endl;
			std::cin.clear(); // clear the error flag
			std::cin.ignore(numeric_limits<streamsize>::max(), '\n');
			std::cout << "Option: ";
			std::cin >> sub_option; // get input
			std::cout << "------------------------------" << std::endl;
		}

		histogram_state = sub_option;
	}

	// show month histogram
	if (options == 4 || options == 8) {
		histogram_mode = 2;
		std::cout << "Choose a month: " << std::endl;
		for (int i = 1; i < 13; i++) {
			std::cout << i << ". " << months[i - 1] << std::endl;
		}
		std::cout << "------------------------------" << std::endl;
		std::cout << "Option: ";
		std::cin.clear();
		std::cin >> sub_option;
		std::cout << "------------------------------" << std::endl;

		// if invalid input
		while (sub_option < 1 || sub_option > 12 || std::cin.fail()) {
			std::cout << "Invalid input! Try again!" << std::endl;
			std::cin.clear(); // clear the error flag
			std::cin.ignore(numeric_limits<streamsize>::max(), '\n');
			std::cout << "Option: ";
			std::cin >> sub_option; // get input
			std::cout << "------------------------------" << std::endl;
		}

		histogram_month = sub_option;
	}
}

std::vector<float> updateHistogram(Weather data) {
	std::vector<float> temperatures;
	std::vector<float> t = data.GetTemp();
	std::vector<string> s = data.GetName();
	std::vector<int> m = data.GetMonth();
	string str = "";

	if (histogram_mode == 0) { // get summary vec
		temperatures = t;
		str = "OVERALL";
	}
	else if (histogram_mode == 1) { // get specific state vec
		//loop through every temperature data
		for (int i = 0; i < t.size(); i++) {
			if (s[i] == states[histogram_state - 1]) { // if is the state data
				temperatures.push_back(t[i]);
			}
		}
		str = states[histogram_state - 1];
	}
	else if (histogram_mode == 2) { // get specific month vec
		//loop through every temperature data
		for (int i = 0; i < t.size(); i++) {
			if (m[i] == histogram_month) { // if is the month data
				temperatures.push_back(t[i]);
			}
		}
		str = months[histogram_month - 1];
	}

	std::cout << "-----------------" << str << " HISTOGRAM (" << method << ")--------------" << std::endl;
	return temperatures;
}

void Serial(std::vector<myType>& Values, bool summary) {
	float startTime = clock();

	std::vector<myType> temperature = Values;

	//Get the size of the vector
	int Size = temperature.size();
	//Perform selection sort
	selectionSort(temperature);
	//Calculate the Min
	float MIN = temperature[0];
	//Calculate the Max
	float MAX = temperature[Size - 1];
	//Calculate the Sum
	float Sum = SStats.Sum(temperature);
	//Calculate the mean
	float Mean = Sum / (Size);
	//Calculate the Standard Deviation
	float SD = SStats.StandardDeviation(temperature);
	//Calculate the Median
	float Median = SStats.GetMedianValue(temperature);
	//Calculate the First Quartile
	float FQuartile = SStats.FirstQuartile(temperature);
	//Calculate the Third Quartile
	float TQuartile = SStats.ThirdQuartile(temperature);

	float endTime = clock();

	//Display header and then all the calculated values
	if (summary) { // if is summary data
		std::cout << "-----------------------OVERALL RESULT (SERIAL)-----------------------" << std::endl;
		// Display in multiple rows
		std::cout << "Records: \t" << Size << std::endl;
		std::cout << "Min: \t\t" << MIN << std::endl;
		std::cout << "Max: \t\t" << MAX << std::endl;
		std::cout << "Mean: \t\t" << Mean << std::endl;
		std::cout << "SD: \t\t" << SD << std::endl;
		std::cout << "Median: \t" << Median << std::endl;
		std::cout << "1Q: \t\t" << FQuartile << std::endl;
		std::cout << "3Q: \t\t" << TQuartile << std::endl;
		std::cout << "Time taken: \t" << (endTime - startTime) << " ms" << std::endl;
	}
	else {
		// Display in one row
		std::cout << Size << "\t";
		std::cout << MIN << "\t";
		std::cout << MAX << "\t";
		std::cout << Mean << "\t";
		std::cout << SD << "\t";
		std::cout << Median << "\t";
		std::cout << FQuartile << "\t";
		std::cout << TQuartile << "\t";
		std::cout << (endTime - startTime) << " ms" << std::endl;
	}
}

void selectionSort(std::vector<myType>& Values) {
	// Time Complexity: O(n2), Space Complexity : O(1)
	int min_idx;
	//One by one move boundary of unsorted array
	for (int i = 0; i < Values.size() - 1; i++) {
		min_idx = i;
		for (int j = i + 1; j < Values.size(); j++) {
			//Store the index of minimum element in unsorted array
			if (Values[j] < Values[min_idx]) {
				min_idx = j;
			}
		}
		//Swap the found minimum element with the first element
		if (min_idx != i) {
			//swapping
			myType temp = Values[min_idx];
			Values[min_idx] = Values[i];
			Values[i] = temp;
		}
	}
}

void SerialSplitStates(std::vector<float>& temp, std::vector<string>& stateName) {
	float startTime = clock();

	// Display the results based on the Five individual States
	std::cout << "-----------------------STATES RESULT (SERIAL)-----------------------" << std::endl;
	std::cout << "STATE    \tRECORDS MIN \tMAX \tMEAN \tSD \tMEDIAN \tQ1 \tQ3 \tTIME" << std::endl;

	// Part of temparatures belong to a specific state
std:vector<float> partTemp;
	for (int i = 0; i < temp.size(); i++) {
		//Is an empty vector
		if (partTemp.size() == 0) {
			partTemp.insert(partTemp.begin(), temp[i]);
		}
		else {
			// Check if the next state is matched with the current state
			if ((i + 1) != temp.size()) { // if not last data
				// if matched, continue adding the temparature to partTemp
				if (stateName[i] == stateName[i + 1]) {
					partTemp.insert(partTemp.begin(), temp[i]);
				}
				else {
					partTemp.insert(partTemp.begin(), temp[i]);
					std::cout << stateName[i] << "  \t";
					Serial(partTemp, false);
					partTemp.clear(); // Reset the partTemp
				}
			}
			else {
				// Last weather data
				partTemp.insert(partTemp.begin(), temp[i]);
				std::cout << stateName[i] << " \t\t";
				Serial(partTemp, false);
				partTemp.clear(); // Reset the partTemp
			}
		}
	}

	float endTime = clock();
	std::cout << "TOTAL TIME: \t" << (endTime - startTime) << " ms" << std::endl;
}

void SerialSplitMonths(std::vector<float>& temp, std::vector<int>& month) {
	float startTime = clock();

	// Display the results based on the twelve individual months
	std::cout << "-----------------------MONTHS RESULT (SERIAL)-----------------------" << std::endl;
	std::cout << "MONTH    \tRECORDS MIN \tMAX \tMEAN \tSD \tMEDIAN \tQ1 \tQ3 \tTIME" << std::endl;

	//Part of temperature belong to a specific month
	std::vector<std::vector<float>> temp2D(12);

	//Loop through all the temperature data
	for (int i = 0; i < temp.size(); i++) {
		temp2D[month[i] - 1].insert(temp2D[month[i] - 1].begin(), temp[i]);
	}
	//Loop through all the month vector
	for (int i = 0; i < 12; i++) {
		std::cout << months[i] << "\t\t";
		Serial(temp2D[i], false);
	}

	float endTime = clock();
	std::cout << "TOTAL TIME: \t" << (endTime - startTime) << " ms" << std::endl;
}

void Parallel(std::vector<float>& Values, cl::Context context, cl::CommandQueue queue, cl::Program program, cl::Event& prof_event, bool summary)
{
	float startTime = clock();
	std::vector<float> temperature = Values;

	//Get the size of the vector
	int Size = temperature.size();
	//Sort the vector into ascending
	Sort(temperature, context, queue, program, prof_event);
	//Calculate the Sum
	float Sum = SumVec(temperature, context, queue, program, prof_event);
	//Calculate the mean
	float Mean = Sum / (Size);
	//Calculate the standard deviation
	float SD = STDVec(temperature, Mean, context, queue, program, prof_event);
	//Calculate the Min
	float MIN = temperature[0];
	//Calculate the Max
	float MAX = temperature[Size - 1];
	//Calculate the Median
	float Median = SStats.GetMedianValue(temperature);
	//Calculate the First Quartile
	float FQuartile = SStats.FirstQuartile(temperature);
	//Calculate the Third Quartile
	float TQuartile = SStats.ThirdQuartile(temperature);

	float endTime = clock();

	//Display header and then all the calculated values
	if (summary) { // if is summary data
		std::cout << "-----------------------OVERALL RESULT (PARALLEL)-----------------------" << std::endl;
		// Display in multiple rows
		std::cout << "Records: \t" << Size << std::endl;
		std::cout << "Min: \t\t" << MIN << std::endl;
		std::cout << "Max: \t\t" << MAX << std::endl;
		std::cout << "Mean: \t\t" << Mean << std::endl;
		std::cout << "SD: \t\t" << SD << std::endl;
		std::cout << "Median: \t" << Median << std::endl;
		std::cout << "Q1: \t\t" << FQuartile << std::endl;
		std::cout << "Q3: \t\t" << TQuartile << std::endl;
		std::cout << "Time taken: \t" << (endTime - startTime) << " ms" << std::endl;
	}
	else {
		// Display in one row
		std::cout << Size << "\t";
		std::cout << MIN << "\t";
		std::cout << MAX << "\t";
		std::cout << Mean << "\t";
		std::cout << SD << "\t";
		std::cout << Median << "\t";
		std::cout << FQuartile << "\t";
		std::cout << TQuartile << "\t";
		std::cout << (endTime - startTime) << " ms" << std::endl;
	}
}

void ParallelSplitStates(std::vector<float>& temp, cl::Context context, cl::CommandQueue queue, cl::Program program, cl::Event& prof_event, std::vector<string>& stateName) {
	float startTime = clock();

	// Display the results based on the Five individual States
	std::cout << "-----------------------STATES RESULT (PARALLEL)-----------------------" << std::endl;
	std::cout << "STATE    \tRECORDS MIN \tMAX \tMEAN \tSD \tMEDIAN \t1Q \t3Q \tTIME" << std::endl;

	// Part of temparatures belong to a specific state
std:vector<float> partTemp;
	for (int i = 0; i < temp.size(); i++) {
		if (partTemp.size() == 0) {
			partTemp.insert(partTemp.begin(), temp[i]);
		}
		else {
			// Check if the next state is matched with the current state
			if ((i + 1) != temp.size()) { // if not last data
				// if matched, continue adding the temparature to partTemp
				if (stateName[i] == stateName[i + 1]) {
					partTemp.insert(partTemp.begin(), temp[i]);
				}
				else {
					partTemp.insert(partTemp.begin(), temp[i]);
					std::cout << stateName[i] << "  \t";
					Parallel(partTemp, context, queue, program, prof_event, false);
					partTemp.clear(); // Reset the partTemp
				}
			}
			else {
				// Last weather data
				partTemp.insert(partTemp.begin(), temp[i]);
				std::cout << stateName[i] << " \t\t";
				Parallel(partTemp, context, queue, program, prof_event, false);
				partTemp.clear(); // Reset the partTemp
			}
		}
	}

	float endTime = clock();
	std::cout << "TOTAL TIME: \t" << (endTime - startTime) << " ms" << std::endl;
}

void ParallelSplitMonths(std::vector<float>& temp, cl::Context context, cl::CommandQueue queue, cl::Program program, cl::Event& prof_event, std::vector<int>& month) {
	float startTime = clock();

	// Display the results based on the twelve individual months
	std::cout << "-----------------------MONTHS RESULT (PARALLEL)-----------------------" << std::endl;
	std::cout << "MONTH    \tRECORDS MIN \tMAX \tMEAN \tSD \tMEDIAN \tQ1 \tQ3 \tTIME" << std::endl;

	//Part of temperature belong to a specific month
	std::vector<std::vector<float>> temp2D(12);

	//Loop through all the temperature data
	for (int i = 0; i < temp.size(); i++) {
		temp2D[month[i] - 1].insert(temp2D[month[i] - 1].begin(), temp[i]);
	}
	//Loop through all the month vector
	for (int i = 0; i < 12; i++) {
		std::cout << months[i] << "\t\t";
		Parallel(temp2D[i], context, queue, program, prof_event, false);
	}

	float endTime = clock();
	std::cout << "TOTAL TIME: \t" << (endTime - startTime) << " ms" << std::endl;
}

//Sum Vector function
myType SumVec(std::vector<myType>& temp, cl::Context context, cl::CommandQueue queue, cl::Program program, cl::Event& prof_event)
{
	//Set local size to 2
	size_t local_size = 2;
	//Add padding to the vector
	int padding_size = AddPadding(temp, local_size, 0.0f);
	//Set kernal to the reduce addition kernal
	cl::Kernel kernel = cl::Kernel(program, "reduce_add_4");
	//Set return to the output from kernal execution
	float Return = KernalExecRet(kernel, temp, local_size, context, queue, true, false, false, 0.0f, 0, prof_event, "Sum Vector");
	//Return value
	return Return;
}

//Standard deviation function
myType STDVec(std::vector<myType>& temp, myType Mean, cl::Context context, cl::CommandQueue queue, cl::Program program, cl::Event& prof_event)
{
	//Get the size of the vector
	double Size = temp.size();
	//Set local size to 2
	size_t local_size = 2;
	//Add padding to the vector
	int padding_size = AddPadding(temp, local_size, 0.0f);
	//Set kernal to the reduce standard deviation kernal
	cl::Kernel kernel = cl::Kernel(program, "reduce_STD_4");
	//Set return to the output from kernal execution
	float Return = KernalExecRet(kernel, temp, local_size, context, queue, true, true, true, Mean, padding_size, prof_event, "Standard Deviation");
	//Divide the result by the size
	Return = (Return / (Size));
	//Square root the result
	Return = sqrt(Return);
	//Return the value
	return Return;
}

//Sort function
void Sort(std::vector<myType>& temp, cl::Context context, cl::CommandQueue queue, cl::Program program, cl::Event& prof_event)
{
	//Set local size to 32
	size_t local_size = (32);
	//Add padding to the vector
	int padding_size = AddPadding(temp, local_size, -1000000.0f);
	//Set kernal to the parallel selection kernal
	cl::Kernel kernel = cl::Kernel(program, "ParallelSelection");
	//Perform the kernal
	KernalExec(kernel, temp, local_size, context, queue, false, false, false, 0.0f, 0, prof_event, "Parallel Selection Sort");
	//Erase the padded elements at the start of the vector
	temp.erase(temp.begin(), temp.begin() + (local_size - padding_size));
}

//Function to add padding to an array
int AddPadding(std::vector<myType>& temp, size_t LocalSize, float PadVal)
{
	//Set the local size
	size_t local_size = LocalSize;
	//Find the padding size
	int padding_size = temp.size() % local_size;
	//If there is padding size then
	if (padding_size)
	{
		//Create an extra vector with PadVal values
		std::vector<float> A_ext(local_size - padding_size, PadVal);
		//Append that extra vector to the input
		temp.insert(temp.end(), A_ext.begin(), A_ext.end());
	}
	//Return padding_size
	return padding_size;
}

//KernalExec is for kernals where the temp vector needs to be overitten
void KernalExec(cl::Kernel kernel, std::vector<myType>& temp, size_t Local_Size, cl::Context context, cl::CommandQueue queue, bool Two, bool Three, bool Four, float FThree, int IFour, cl::Event& prof_event, std::string Name)
{
	//Get the size of the vector
	double Size = temp.size();

	//Get the number of input elements
	size_t input_elements = temp.size();
	//Size in bytes of the input vector
	size_t input_size = temp.size() * sizeof(myType);

	//Define Output vector B
	std::vector<myType> B(input_elements);
	//Get the size in bytes of the output vector
	size_t output_size = B.size() * sizeof(myType);

	//Setup device buffer
	cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

	//Write all the values from temp into the buffer
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &temp[0], NULL, &prof_event);
	queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);

	//Set the arguments 0 and 1 to be the buffers
	kernel.setArg(0, buffer_A);
	kernel.setArg(1, buffer_B);

	//If two is true then set argument two to the local memory size
	if (Two == true)
		kernel.setArg(2, cl::Local(Local_Size * sizeof(myType)));//Local memory size
	//If three is true then set argument three to the float value passed into the function
	if (Three == true)
		kernel.setArg(3, FThree);
	//If four is true then set argument three to the int value passed into the function
	if (Four == true)
		kernel.setArg(4, IFour);

	//Run the kernal
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(Local_Size), NULL, &prof_event);

	//Copy the result from device to host
	//Setup prof Event
	cl::Event prof_event2;
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &temp[0], NULL, &prof_event2);
}

//KernalExecRet is for kernals where the temp vector does not need to be overitten, but the first element of B returned
float KernalExecRet(cl::Kernel kernel, std::vector<myType>& temp, size_t Local_Size, cl::Context context, cl::CommandQueue queue, bool Two, bool Three, bool Four, float FThree, int IFour, cl::Event& prof_event, std::string Name)
{
	//Get the size of the vector
	double Size = temp.size();

	//Get the number of input elements
	size_t input_elements = temp.size();
	//Size in bytes of the input vector
	size_t input_size = temp.size() * sizeof(myType);

	//Define Output vector B
	std::vector<myType> B(input_elements);
	//Get the size in bytes of the output vector
	size_t output_size = B.size() * sizeof(myType);

	//Setup device buffer
	cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

	//Set the arguments 0 and 1 to be the buffers
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &temp[0], NULL, &prof_event);

	queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);

	//Set the arguments 0 and 1 to be the buffers
	kernel.setArg(0, buffer_A);
	kernel.setArg(1, buffer_B);

	//If two is true then set argument two to the local memory size
	if (Two == true)
		kernel.setArg(2, cl::Local(Local_Size * sizeof(myType)));//Local memory size
	//If three is true then set argument three to the float value passed into the function
	if (Three == true)
		kernel.setArg(3, FThree);
	//If four is true then set argument three to the int value passed into the function
	if (Four == true)
		kernel.setArg(4, IFour);

	//Run the kernal
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(Local_Size), NULL, &prof_event);

	//Copy the result from device to host
	//Setup prof Event
	cl::Event prof_event2;
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0], NULL, &prof_event2);

	//Display kernal memory read time
	//std::cout << Name << " Kernel memory read time [ns]:" << prof_event2.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event2.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	//std::cout << GetFullProfilingInfo(prof_event2, ProfilingResolution::PROF_US) << std::endl << std::endl;

	//Return the first element of the buffer Vector B
	return B[0];
}

void serialHistogram(std::vector<float>& temperature, float minimum, float maximum) {
	//Create output vector
	std::vector<int> histogram_vector(histogram_bin_no); // histogram results

	//display bins and frequency
	std::cout << "Minimum: " << minimum << ", Maximum: " << maximum << std::endl;
	std::cout << "Number of Bins: " << histogram_bin_no << ", Bin Size: " << (maximum - minimum) / histogram_bin_no << std::endl;
	float binSize = (maximum - minimum) / histogram_bin_no;
	int max_freq = 0;

	// clean vector
	upperLimits.clear();
	frequencies.clear();

	upperLimits.push_back(minimum); // first element is the minimum of elements

	for (int i = 0; i < temperature.size(); i++) {
		float compareVal = minimum + binSize;
		int idx = 0;
		while (temperature[i] > compareVal) {
			compareVal += binSize; // check next range
			idx++;
		}
		if (idx == histogram_bin_no) {
			idx--;
		}
		histogram_vector[idx] += 1;
	}

	for (int i = 1; i < histogram_bin_no + 1; i++) {
		float binStart = minimum + ((i - 1) * binSize);
		float binEnd = minimum + (i * binSize);
		int frequency = (histogram_vector[i - 1]);
		std::cout << "Bin Range: >" << binStart << " to <=" << binEnd << ", Frequency: " << frequency << std::endl;

		max_freq = (frequency > max_freq) ? frequency : max_freq;
		frequencies.push_back(frequency);
		upperLimits.push_back(binEnd);
	}

	frequencies.push_back(max_freq); // last element is the total number of frequencies
}

// Histogram implementation (in parallel)
void parallelHistogram(std::vector<float>& temperature, cl::Context context, cl::CommandQueue queue, cl::Program program, cl::Event& prof_event, float minimum, float maximum) {
	cl::Kernel kernel(program, "histogram");

	//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
	//if the total input length is divisible by the workgroup size
	//this makes the code more efficient
	size_t local_size = 256; //1024; //work group size - higher work group size can reduce 
	size_t padding_size = temperature.size() % local_size;

	//if the input vector is not a multiple of the local_size
	//insert additional neutral elements (0 for addition) so that the total will not be affected (make work for my working set of data)
	if (padding_size) {
		//create an extra vector with neutral values
		std::vector<int> temperature_ext(local_size - padding_size, 1000);
		//append that extra vector to our input
		temperature.insert(temperature.end(), temperature_ext.begin(), temperature_ext.end());
	}

	size_t vector_elements = temperature.size();//number of elements
	size_t vector_size = temperature.size() * sizeof(int);//size in bytes

	//Create output vector
	vector<int> histogram_vector(histogram_bin_no); // histogram results
	vector<int> output(histogram_vector.size());
	size_t output_size = output.size() * sizeof(float);

	//Create buffers
	cl::Buffer input_buffer(context, CL_MEM_READ_WRITE, vector_size);
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, output_size);

	//Create queue and copy vectors to device memory
	queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, vector_size, &temperature[0]);
	queue.enqueueFillBuffer(output_buffer, 0, 0, output_size);//zero B buffer on device memory

	//Set the arguments 0 and 3 to be the buffers
	kernel.setArg(0, input_buffer);
	kernel.setArg(1, output_buffer);
	kernel.setArg(2, histogram_bin_no);
	kernel.setArg(3, minimum);
	kernel.setArg(4, maximum);

	//Execute kernel
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));

	//Copy the result from device to host
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, output_size, &histogram_vector[0]);

	//display bins and frequency
	std::cout << "Minimum: " << minimum << ", Maximum: " << maximum << std::endl;
	std::cout << "Number of Bins: " << histogram_bin_no << ", Bin Size: " << (maximum - minimum) / histogram_bin_no << std::endl;
	float binSize = (maximum - minimum) / histogram_bin_no;
	int max_freq = 0;

	// clean vector
	upperLimits.clear();
	frequencies.clear();

	upperLimits.push_back(minimum); // first element is the minimum of elements

	for (int i = 1; i < histogram_bin_no + 1; i++) {
		float binStart = minimum + (i - 1) * binSize;
		float binEnd = minimum + i * binSize;
		int frequency = (histogram_vector[i - 1]);
		std::cout << "Bin Range: >" << binStart << " to <=" << binEnd << ", Frequency: " << frequency << std::endl;

		max_freq = (frequency > max_freq) ? frequency : max_freq;
		frequencies.push_back(frequency);
		upperLimits.push_back(binEnd);
	}

	frequencies.push_back(max_freq); // last element is the total number of frequencies
}

void initHistogram(void) {
	glClearColor(1.0, 1.0, 1.0, 0.0); // init the background color to white
}

void reshapeWindow(int w, int h) {
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);							// viewport
	glLoadIdentity();										// resize the projection matrix
	//simple 2D cartesian plane, only xand y
	gluOrtho2D(0, histogram_wwidth, 0, histogram_wheight);	// switch the plane of origin at bottom left
	glMatrixMode(GL_MODELVIEW);
}

void renderHistogram(void) {
	glClear(GL_COLOR_BUFFER_BIT); // clean the screen of all previous drawing
	glLoadIdentity();

	int margin = 50;
	int padding = 10;
	int x1 = margin;
	int x2 = histogram_wwidth - margin;
	int y1 = margin;
	int y2 = histogram_wheight - margin;
	float lw = (float)(x2 - margin) / (float)histogram_bin_no;
	string text;

	glColor3f(0.0, 0.6, 0.9);
	//draw each bar chart
	for (int i = 0; i < histogram_bin_no; i++) {

		text = to_string(frequencies[i]);
		float rx = margin + (i * lw);
		float ry = margin;
		float rw = (float)((histogram_wwidth - (2 * margin)) / histogram_bin_no);
		float rh = ((float)frequencies[i] / (float)frequencies[histogram_bin_no]) * (histogram_wheight - (2 * margin) - padding);
		float tx = rx + (float)(lw / (text.size() + 1));
		float ty = rh + margin + padding;

		drawRect(rx, ry, rw, rh);
		drawText(text.data(), text.size(), tx, ty, GLUT_BITMAP_HELVETICA_10);
	}

	float tx = margin;
	float ty = histogram_wheight - (margin / 2);

	if (histogram_mode == 0) {
		text = "Histogram: OVERALL";
	}
	else if (histogram_mode == 1) { // show state histogram
		text = "Histogram: " + states[histogram_state - 1];
	}
	else if (histogram_mode == 2) { // show month histogram
		text = "Histogram: " + months[histogram_month - 1];
	}

	glColor3f(0.2, 0.2, 0.2);
	drawText(text.data(), text.size(), tx, ty, GLUT_BITMAP_HELVETICA_18); // title

	drawLine(x1, y1, x1, y2); // y-axis
	drawLine(x1, y1, x2, y1); // x-axis
	float idx = 0;
	// x-axis marker and label
	for (float i = x1; i <= x2; i += lw) {
		float lx = i + lw;

		std::stringstream stream;
		stream << std::fixed << std::setprecision(2) << upperLimits[idx];
		string text = stream.str();

		drawLine(lx, y1 + 5, lx, y1 - 5);
		drawText(text.data(), text.size(), i - 5, y1 - 15, GLUT_BITMAP_HELVETICA_10);
		idx++;
	}

	glFlush();
}

void drawLine(GLint x1, GLint y1, GLint x2, GLint y2) {
	// increase the width of the line
	glLineWidth(2);
	// draw the axis
	glBegin(GL_LINES);
	glColor3f(0.2, 0.2, 0.2);
	glVertex2f(x1, y1);
	glVertex2f(x2, y2);
	glEnd();

	glFlush();
}

void drawRect(GLint x, GLint y, GLint width, GLint height) {
	glPushMatrix();
	// start the drawing of vertices
	glBegin(GL_QUADS); // quadrilateral
	// points should be anticlockwise to show the front side of polygon
	glVertex2f(x, y);
	glVertex2f(x + width, y);
	glVertex2f(x + width, y + height);
	glVertex2f(x, y + height);

	// end of vertices
	glEnd();
	glPopMatrix();
}

void drawText(const char* text, GLint length, GLint x, GLint y, void* font) {
	double* matrix = new double[16];
	glGetDoublev(GL_PROJECTION_MATRIX, matrix);

	glPushMatrix();
	glLoadIdentity();
	glRasterPos2i(x, y);
	for (int i = 0; i < length; i++) {
		glutBitmapCharacter(font, (int)text[i]);
	}
	glPopMatrix();

	glMatrixMode(GL_PROJECTION);
	glLoadMatrixd(matrix);
	glMatrixMode(GL_MODELVIEW);
}