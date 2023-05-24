#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <wiringPi.h>
#include <wiringSerial.h>
#include <sys/time.h>
#include <math.h>
#include "MATLAB_generated_files/rtwtypes.h"
#include <stddef.h>

#ifndef AM7_H
#define AM7_H

//Define the baudrate for the module and the starting byte 
#define START_BYTE 0x9B  //1st start block identifier byte
#define BAUDRATE_AM7 460800 //Define the baudrate
#define MAX_FREQUENCY_MSG_OUT 300 //Define the maximum message output frequency

//Communication structures
struct  __attribute__((__packed__)) am7_data_out {
	int16_T motor_1_cmd_int;
	int16_T motor_2_cmd_int;
	int16_T motor_3_cmd_int;
	int16_T motor_4_cmd_int;
	int16_T el_1_cmd_int;
	int16_T el_2_cmd_int;
	int16_T el_3_cmd_int;
    int16_T el_4_cmd_int;
    int16_T az_1_cmd_int;
    int16_T az_2_cmd_int;
    int16_T az_3_cmd_int;
    int16_T az_4_cmd_int;
	int16_T theta_cmd_int;
    int16_T phi_cmd_int;
	//Optimization info
    uint16_T n_iteration;
    uint16_T n_evaluation;
	uint16_T elapsed_time_us;
    int16_T exit_flag_optimizer;
    //Residuals
    int16_T residual_ax_int;
    int16_T residual_ay_int;
    int16_T residual_az_int;
    int16_T residual_p_dot_int;
    int16_T residual_q_dot_int;
    int16_T residual_r_dot_int;
    float rolling_msg_out;
    uint8_T rolling_msg_out_id;
	uint8_T checksum_out;
};

struct  __attribute__((__packed__)) am7_data_in {
    //Actuator state
    int16_T motor_1_state_int;
    int16_T motor_2_state_int;
    int16_T motor_3_state_int;
    int16_T motor_4_state_int;
    int16_T el_1_state_int;
    int16_T el_2_state_int;
    int16_T el_3_state_int;
    int16_T el_4_state_int;
    int16_T az_1_state_int;
    int16_T az_2_state_int;
    int16_T az_3_state_int;
    int16_T az_4_state_int;
    //Variable states
    int16_T theta_state_int;
    int16_T phi_state_int;
    int16_T psi_state_int;
    int16_T gamma_state_int;
    int16_T p_state_int;
    int16_T q_state_int;
    int16_T r_state_int;
    int16_T airspeed_state_int;
    //Pseudo-control cmd
    int16_T pseudo_control_ax_int;
    int16_T pseudo_control_ay_int;
    int16_T pseudo_control_az_int;
    int16_T pseudo_control_p_dot_int;
    int16_T pseudo_control_q_dot_int;
    int16_T pseudo_control_r_dot_int;
    //Desired actuator value:
    int16_t desired_motor_value_int;
    int16_t desired_el_value_int;
    int16_t desired_az_value_int;
    int16_t desired_theta_value_int;
    int16_t desired_phi_value_int;
    float rolling_msg_in;
    uint8_T rolling_msg_in_id;  
	uint8_T checksum_in;
};

#endif