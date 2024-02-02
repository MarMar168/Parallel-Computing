#pragma once

//Includes
#include <iostream>
#include <string>
#include <vector>

//Sort Type ENUM
enum SORT { ASCENDING, DECENDING };
//Define MyType
typedef float myType;

class SerialStatistics
{
public:
	//Empty Constructor and Destructor
	SerialStatistics();
	~SerialStatistics();
	
	//Sorts a input vector using the bubble sort algorithm
	std::vector<myType>& SerialStatistics::Sort(std::vector<myType>& Values, SORT Mode);
	//Finds the sum of an input vector
	myType SerialStatistics::Sum(std::vector<myType>& Values);
	//Finds the Min or Max of an input vector
	myType SerialStatistics::MinMax(std::vector<myType>& Values, bool MinMax);
	//Displays a vector 
	void SerialStatistics::Display(std::vector<myType>& Values);
	//Gets the median value from an input vector
	myType SerialStatistics::GetMedianValue(std::vector<myType>& Values);
	//Gets the mean value from an input vector
	myType SerialStatistics::Mean(std::vector<myType>& Values);
	//Gets the Standard Deviation value from an input vector
	myType SerialStatistics::StandardDeviation(std::vector<myType>& Values);
	//Gets the First Quartile value from an input vector
	myType SerialStatistics::FirstQuartile(std::vector<myType>& Values);
	//Gets the Third Quartile value from an input vector
	myType SerialStatistics::ThirdQuartile(std::vector<myType>& Values);
};