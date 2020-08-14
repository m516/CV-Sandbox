#include <stdio.h>
#include <assert.h>
#include <iostream>


#include "optical_flow.hpp"

bool OpticalFlow::process()
{
	//Don't process empty data
	if (d_imageInputData.empty()) return false;
	//Do bogus stuff for now
	d_imageInputData.copyTo(d_imageOutputData);
	alreadyProcessed = true;
	return true;
}
