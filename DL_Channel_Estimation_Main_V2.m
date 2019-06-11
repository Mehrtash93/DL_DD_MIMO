clc
clear all

snr_all = 2:4:30;
for i = 1:length(snr_all)

NT=2;
NR=2;                                                                                                                                                                 
Ts=10^-5 ;
fd=2000;
fd_miss = 500+fd;

Pilot_length=12;
SNR=snr_all(i);
% SNR = 20;
Mont_Carlo=10000;
Training_number=10000;

%********BPSKModulation**********
% bpskModulator = comm.BPSKModulator;
% bpskModulator.PhaseOffset = 0;
% alphabet=[bpskModulator(0) bpskModulator(1)];
% I=repmat(alphabet,NT,1);
% M=2;
%********************************

M=4;                                      %Modulation order 
DD = (0:M-1)';
alphabet=transpose(qammod(DD,M));          
I=repmat(alphabet,NT,1);


SNR = SNR + 10 * log10(M);

[Input1,Input2,Output1,Output2,IN_V1,IN_V2,OUT_V1,OUT_V2]=DL_Channel_Estimation_Training_V2(NT,NR,Ts,fd_miss,Pilot_length,SNR,alphabet,Training_number);
% save('Train_Channel_100_11_22x_16QAM','Input','Output')
% load Train_Channel_100_11_22x_16QAM


options1 = trainingOptions('adam','MaxEpochs',2000,'InitialLearnRate',0.001,'MiniBatchSize',10,'Shuffle','every','L2Regularization',0,'Plots','training-progress','ValidationData',{IN_V1,OUT_V1})
options2 = trainingOptions('adam','MaxEpochs',2000,'InitialLearnRate',0.001,'MiniBatchSize',10,'Shuffle','every','L2Regularization',0,'Plots','training-progress','ValidationData',{IN_V2,OUT_V2})
%*********************************************************************************************************************************************************************************
layers =[sequenceInputLayer([NT*NR*Pilot_length]) fullyConnectedLayer(128) clippedReluLayer(1) fullyConnectedLayer(128) clippedReluLayer(1)  ...
fullyConnectedLayer(2*NT*NR) regressionLayer];
Net1 = trainNetwork(Input1(:,1:Training_number-1000),Output1(:,1:Training_number-1000),layers,options1);                  %Training the Network real
Net2 = trainNetwork(Input2(:,1:Training_number-1000),Output2(:,1:Training_number-1000),layers,options2);                  %Training the Network imaginary

rate = .1:.025:.5;
Packets = 60;
for j=1:length(Packets)
Packet_Length=Packets(j);
fd = 1000*rand()+fd;
for m=1:Mont_Carlo
m
[A(m),B(m),C(m),y_origin,y_MMSE,y_DL] = Decision_Directed_Channel_Estimation_V2(NT,NR,Ts,fd,Packet_Length,Pilot_length,SNR,alphabet,Net1,Net2);
end

BER_MMSE(i) = mean(A)
BER_DL(i) = mean(B)
BER_Orig(i) = mean(C)
end
end
save('22_fd_max2000_packet_missmatch','BER_MMSE','BER_DL','y_origin','y_MMSE','y_DL')

