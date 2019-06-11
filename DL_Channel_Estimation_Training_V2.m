function [Input_T1,Input_T2,Output_T1,Output_T2,Input_V1,Input_V2,Output_V1,Output_V2] = DL_Channel_Estimation_Training_V2(NT,NR,Ts,fd,Pilot_length,SNR,alphabet,Training_number)

warning('off') 
E1 = mean(abs(alphabet).^2);
sigma_w = (E1*NT) * (10^(-SNR/10));          %Noise variance



for kx=1:Training_number
    kx
Channel= rayleighchan(Ts,fd);
for k1 = 1:NR
    for k2=1:NT
        H_training(k2,:) = filter(Channel, ones(1,Pilot_length+2));
    end
    HH(:,:,k1)=H_training;
end
S=[];
for k3=1:(Pilot_length+2)/2
    S0=randsrc(1,NT,alphabet);
    S1=transpose(alamoutivector(S0));
    S=blkdiag(S,S0);
    S=blkdiag(S,S1);
end

for k4 = 1:NR
    ht=HH(:,:,k4);
    hc=ht(:);
    Hc_training(:,k4)=hc;
    RT(:,k4)=S*hc;
end

% for k10 = NR+1:2*NR
%     ht=HH(:,:,k10);
%     hc=ht(:);
%     hc_new = alamoutivector(hc);
%     Hc_training(:,k10)=hc;
%     RT(:,k10)=conj(S*hc_new);
% end

WT=sqrt(sigma_w/2)*(randn(Pilot_length+2,NR)+1i*randn(Pilot_length+2,NR));
RT=RT+WT;


%for v=1:Pilot_length*NT-NT
H_o=Hc_training(1:Pilot_length*NT,:);
H_oo=H_o(:);
H_r_i=real(H_oo);
H_i_i=imag(H_oo);
%r_new = [RT(end-1,:) RT(end,:)];
%R_I=[real(r_new(:));imag(r_new(:))];
Input_1 = H_r_i;
Input_2 = H_i_i;

    
H_I=Hc_training(end-2*NT+1:end,:);
H_II=H_I(:);
H_r_o=real(H_II);
H_i_o=imag(H_II);
Output_1 = H_r_o;
Output_2 = H_i_o;

%end

Input1(:,:,kx)=Input_1;
Input2(:,:,kx)=Input_2;
Output1(:,:,kx)=Output_1;
Output2(:,:,kx)=Output_2;
    


end
%whos Output_T2
Input_T1=reshape(Input1,NT*NR*Pilot_length,[]);
Input_T2=reshape(Input2,NT*NR*Pilot_length,[]);

Output_T1=reshape(Output1,2*NT*NR,[]);
Output_T2=reshape(Output2,2*NT*NR,[]);

Input_V1 = Input_T1 (:,end-1000+1:end);
Input_V2 = Input_T2 (:,end-1000+1:end);
Output_V1 = Output_T1 (:,end-1000+1:end);
Output_V2 = Output_T2 (:,end-1000+1:end);
Input_T1 = Input_T1 (:,1:end-1000);
Input_T2 = Input_T2 (:,1:end-1000);
Output_T1 = Output_T1 (:,1:end-1000);
Output_T2 = Output_T2 (:,1:end-1000);


