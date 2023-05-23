# Orangepi_PPZ
This repository enables the interfacing of a Single Board Computer (SBC) such as the OrangePi or the Raspberry Pi with an autopilot like the Orange Cube using UART communication.

The primary function in this repository is the "am7.c" function. This function initializes two threads that run in parallel. The first thread is responsible for executing the MATLAB-generated function or any other function relevant to the autopilot. The second thread handles the communication with the Orange Cube using the WiringPi library and a UART communication. It is important to note that for the proper utilization of this repository, the Orangepi_PPZ module should be loaded and executed on Paparazzi UAV.
