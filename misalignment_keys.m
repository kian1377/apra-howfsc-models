clear all; close all; clc

keys = ["decenter_z" "decenter_x" "decenter_y" "tilt_x" "tilt_y" "tilt_z"];
m1_vals = [1e-6 0 1e-4 0 0 0];
m1_vals(1:3) = m1_vals(1:3)*1e3; % change decenters to be in mm
m1_misalignment = dictionary(keys, m1_vals);

m1_misalignment('decenter_y')

