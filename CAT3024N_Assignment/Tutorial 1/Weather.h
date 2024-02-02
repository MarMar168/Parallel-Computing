#pragma once

//Include
#include <vector>
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>

class Weather
{
public:
	//Empty constructor and destructor
	Weather();
	~Weather();
	//Load weather text file
	bool Weather::Load(std::string FileName);

	//Getter functions
	std::vector<float>& GetTemp() { return m_Temp; }
	std::vector<int>& GetTempI() { return m_TempI; }
	std::vector<int>& GetDay() { return m_Day; }
	std::vector<int>& GetYear() { return m_Year; }
	std::vector<int>& GetMonth() { return m_Month; }
	std::vector<int>& GetTime() { return m_Time; }
	std::vector<std::string>& GetName() { return m_Name; }

private:
	//Vector to hold integer temperature values
	std::vector<int> m_TempI;
	//Vector to hold float temperature values
	std::vector<float> m_Temp;
	//Vectors to hold days, years, months, time
	std::vector<int> m_Day;
	std::vector<int> m_Year;
	std::vector<int> m_Month;
	std::vector<int> m_Time;
	//Vector to hold names
	std::vector<std::string> m_Name;
};