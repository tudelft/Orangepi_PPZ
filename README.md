# Orangepi_PPZ
This repository enables the interfacing of a Single Board Computer (SBC) such as the OrangePi or the Raspberry Pi with an autopilot like the Orange Cube using UART communication.

The primary function in this repository is the "am7x.c" function. This function initializes two threads that run in parallel. The first thread is responsible for executing the MATLAB-generated function or any other function relevant to the autopilot. The second thread handles the communication with the Orange Cube using the WiringPi library and a UART communication. It is important to note that for the proper utilization of this repository, the ca_am_7 module should be loaded and executed on Paparazzi UAV.

Note that to compile the functions you need to have the WiringPi and pthread library installed on the SBC. These libraries are installed by default in the OrangePi OS (http://www.orangepi.org/html/softWare/orangePiOS/index.html). Moreover, the UART serial should be enabled in the settings of the OrangePi 5. For this, please refer to the OrangePi 5 manual [OrangePi_5_technical_manual.pdf](Documentation/OrangePi_5_technical_manual.pdf).

To test the code you can clone the repository on the OrangePi 5 SBC and compile it with the provided makefile. 

# Wiring: 
The wirings between the Orange cube and the OrangePi 5 are reported in the following diagram: [OrangePi_PPZ_wirings.pdf](Documentation/OrangePi_PPZ_wirings.pdf).

**Connection to the Autopilot:**
|  AP PIN | OrangePi 5 PIN |
| ----- | -------- |
| AP UART TX | GPIO4_A4 | 
| AP UART RX | GPIO4_A3 | 

For the identification of the orangepi pins, please refer to the OrangePi website: http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-5.html
