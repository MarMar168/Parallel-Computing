#include "Weather.h"

//Empty Constructor
Weather::Weather()
{
}

//Empty Destructor
Weather::~Weather()
{
}

//Load File
bool Weather::Load(std::string FileName)
{
	//Setup Variables to hold information for each line
	std::string place;
	int y, m, d, t;
	float tem;
	//Load the txt file
	std::string info;
	std::ifstream weatherFile;
	//Open the text file
	weatherFile.open(FileName);
	//Display Message
	std::cout << "Starting Loading" << std::endl;
	//Until end of file do the following
	while (!weatherFile.eof())
	{
		//Get the current line of information and assign it to the correct place
		weatherFile >> place >> y >> m >> d >> t >> tem;
		//Push back the current line information into the vectors for storage
		m_Name.push_back(place);
		m_Year.push_back(y);	
		m_Month.push_back(m);
		m_Day.push_back(d);
		m_Time.push_back(t);
		m_Temp.push_back(tem);
		m_TempI.push_back((int)(tem * 10.0f));
	}
	//Display loaded message to console
	std::cout << "File Loaded\n" << std::endl;
	//Return true
	return true;
}