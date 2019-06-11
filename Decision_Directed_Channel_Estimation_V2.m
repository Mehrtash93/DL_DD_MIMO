function [BER_MMSE,BER_DL,BER_Orig,y_origin,y_MMSE,y_DL] = Decision_Directed_Channel_Estimation_V2(NT,NR,Ts,fd,Packet_Length,Pilot_length,SNR,alphabet,Net1,Net2)

warning('off')
E1 = mean(abs(alphabet).^2);
sigma_w = (E1*NT) * (10^(-SNR/10));          %Noise variance
snr=E1/sigma_w;

fd_miss = 1000+fd;
Rh=zeros(1,Pilot_length);
for kb=0:NT:Pilot_length*NT+1
    Rh(kb+1)=besselj(0,2*pi*fd_miss*Ts*kb/NT);
end
RH=toeplitz(Rh);
RH=RH(1:Pilot_length*NT,1:Pilot_length*NT);             %Correlation matrix of fading channel
EYE_RH = eye(Pilot_length*NT);

Channel= rayleighchan(Ts,fd);
for k1 = 1:NR
    for k2=1:NT
        H_t(k2,:) = filter(Channel, ones(1,Packet_Length));
    end
    H(:,:,k1)=H_t;
end
S=[];
for k3=1:(Packet_Length)/2
    S0=randsrc(1,NT,alphabet);
    S1=transpose(alamoutivector(S0));
    S=blkdiag(S,S0);
    S=blkdiag(S,S1);
end

for k4 = 1:NR
    h=H(:,:,k4);
    hc=h(:);
    Hc(:,k4)=hc;
    R(:,k4)=S*hc;
end

% for k4 = NR+1:2*NR
%     h=H(:,:,k4);
%     hc=h(:);
%     hc_new = alamoutivector(hc);
%     Hc(:,k4)=hc_new;
%     R(:,k4)=conj(S*hc_new);
% end
W=sqrt(sigma_w/2)*(randn(Packet_Length,NR)+1i*randn(Packet_Length,NR));

R=R+W;

S_Win_MMSE=S(1:Pilot_length,1:(Pilot_length)*NT);
R_Win_MMSE=R(1:Pilot_length,:);


S_Detected_MMSE_Mat=S_Win_MMSE;
Data_DL_Mat=S_Detected_MMSE_Mat;
Data_Orig_Mat = S_Win_MMSE;
H_DL_Pilot=RH*S_Win_MMSE'/(S_Win_MMSE*RH*S_Win_MMSE'+sigma_w*eye(Pilot_length))*R_Win_MMSE;

S_Detected_MMSE_Mat2 = S_Detected_MMSE_Mat;
S_Win_MMSE2 = S_Win_MMSE;

for k5=Pilot_length+1:2:Packet_Length
    H_hat=RH*S_Win_MMSE'/(S_Win_MMSE*RH*S_Win_MMSE'+sigma_w*eye(Pilot_length))*R_Win_MMSE;
    H_detection= H_hat(end-NT+1:end,:);
    R=R-W;
    r_instant=[R(k5,:)+W(k5,:) conj(R(k5+1,:))+W(k5+1,:)];
    R=R+W;
    Hx=transpose(H_detection);
    Hx_new=[Hx;transpose(-alamoutivector(Hx(1,:)));transpose(-alamoutivector(Hx(2,:)))];
    
%     S_det=((Hx'*Hx+snr^-1*eye(NT))\Hx'*transpose(r_instant));
%     [~,D_mmse]=min(abs(repmat(S_det,1,M)-alphabet),[],2);
%     D=alphabet(D_mmse);
%     S_Detected_MMSE_Mat=blkdiag(S_Detected_MMSE_Mat,D);
    
%     [ii,jj] = size(H_hat);
%     H_hat_i = H_hat;
%     C = 2*NT / sigma_w;
%     for i1 = 1:5
%         H_hat_i = inv(C*eye(Pilot_length*NT)+inv(RH))*((C*eye(Pilot_length*NT)-S_Win_MMSE2'*inv(sigma_w*eye(Pilot_length))*S_Win_MMSE2)*H_hat_i+S_Win_MMSE2'*inv(sigma_w*eye(Pilot_length))*R_Win_MMSE);
%     end
%     H_detection2= H_hat_i(end-NT+1:end,:);
%     Hx2 = transpose(H_detection2);
%     
%     S_det2 = sphdec(Hx2, transpose(r_instant), alphabet, 300);
%     S_Detected_MMSE_Mat2=blkdiag(S_Detected_MMSE_Mat2,transpose(S_det2));
%     S_Win_MMSE2=blkdiag(S_Win_MMSE2(2:end,NT+1:end),transpose(S_det2));
    
    S_det = sphdec(Hx_new, transpose(r_instant), alphabet, 300);
    S_Detected_MMSE_Mat=blkdiag(S_Detected_MMSE_Mat,transpose(S_det));
    S_Detected_MMSE_Mat=blkdiag(S_Detected_MMSE_Mat,transpose(alamoutivector(transpose(S_det))));
    
    S_Win_MMSE=blkdiag(S_Win_MMSE(3:end,2*NT+1:end),transpose(S_det));
    S_Win_MMSE=blkdiag(S_Win_MMSE,transpose(alamoutivector(transpose(S_det))));
    R_Win_MMSE=R(k5-Pilot_length+2:k5+1,:);
    
    %********************************************* DL_Part
    %r_instant2=[R(k5,:) R(k5+1,:)];
  
    H_DL_1=H_DL_Pilot(:);
    H_real_DL=real(H_DL_1);
    H_image_DL=imag(H_DL_1);
    %R_instant=[real(r_instant2),imag(r_instant2)];
    Input_DL_Real = H_real_DL;
    Input_DL_Imaginary = H_image_DL;

    
    H_Estimate_DL_Real = predict(Net1,Input_DL_Real);
    H_Estimate_DL_Imaginary = predict(Net2,Input_DL_Imaginary);
    
    H_Estimate_DL_Complex=H_Estimate_DL_Real + 1i*H_Estimate_DL_Imaginary;
    H_Estimate_DL_Complex_Mat_1=reshape(H_Estimate_DL_Complex,2*NT,NR);
    H_Estimate_DL_Complex_Mat_22=transpose(H_Estimate_DL_Complex_Mat_1);
    H_Estimate_DL_Complex_Mat_2=[H_Estimate_DL_Complex_Mat_22(:,1:2);H_Estimate_DL_Complex_Mat_22(:,3:4)];
    
%     MMSE_DL=((H_Estimate_DL_Complex_Mat_2'*H_Estimate_DL_Complex_Mat_2+snr^-1*eye(NT))\H_Estimate_DL_Complex_Mat_2'*transpose(r_instant));
%     [~,Decision_DL_Index]=min(abs(repmat(MMSE_DL,1,M)-alphabet),[],2);
%     Decision_DL=alphabet(Decision_DL_Index);
%     Data_DL_Mat=blkdiag(Data_DL_Mat,Decision_DL);
    
    H_Estimate_DL_Complex_Mat_3 = [H_Estimate_DL_Complex_Mat_2(1:2,:);transpose(-alamoutivector(H_Estimate_DL_Complex_Mat_2(3,:)));transpose(-alamoutivector(H_Estimate_DL_Complex_Mat_2(4,:)))];
    Decision_DL = sphdec(H_Estimate_DL_Complex_Mat_3, transpose(r_instant), alphabet, 300);

 
    Data_DL_Mat=blkdiag(Data_DL_Mat,transpose(Decision_DL));
    Data_DL_Mat=blkdiag(Data_DL_Mat,transpose(alamoutivector(transpose(Decision_DL))));


    S_LS=Data_DL_Mat(end-Pilot_length+1:end,end-NT*Pilot_length+1:end);
    
    H_DL_Pilot=EYE_RH*S_LS'/(S_LS*EYE_RH*S_LS'+sigma_w*eye(Pilot_length))*R_Win_MMSE;
    %H_DL_Pilot=S_LS'*inv(S_LS*S_LS')*R_Win_MMSE;
    
    
    %H_DL_Pilot=[H_DL_Pilot(NT+1:end,:);H_Estimate_DL_Complex_Mat_3];
    
    H_original_1 = transpose(Hc(NT*(k5-Pilot_length-1)+Pilot_length*NT+1:NT*(k5-Pilot_length)+Pilot_length*NT,:));
    H_original_2 = transpose(Hc(NT*(k5+1-Pilot_length-1)+Pilot_length*NT+1:NT*(k5+1-Pilot_length)+Pilot_length*NT,:));
    H_original = [H_original_1;transpose(-alamoutivector(H_original_2(1,:)));transpose(-alamoutivector(H_original_2(2,:)))];
    Optimal_Case = sphdec(H_original, transpose(r_instant), alphabet, 300);
    Data_Orig_Mat=blkdiag(Data_Orig_Mat,transpose(Optimal_Case));
    Data_Orig_Mat=blkdiag(Data_Orig_Mat,transpose(alamoutivector(transpose(Optimal_Case))));
   
    y_origin(k5-Pilot_length) = H_original(1,1);
    y_MMSE(k5-Pilot_length) = Hx_new(1,1);
    y_DL(k5-Pilot_length) = H_Estimate_DL_Complex_Mat_3(1,1);
    
end
 
Detection_MMSE_Error=S-S_Detected_MMSE_Mat;
BER_MMSE=sum(sum(Detection_MMSE_Error~=0))/(Packet_Length*NT);

% Detection_MMSE_Error2=S-S_Detected_MMSE_Mat2;
% BER_MMSE2=sum(sum(Detection_MMSE_Error2~=0))/(Packet_Length*NT);

Detection_DL_Error=S-Data_DL_Mat;
BER_DL=sum(sum(Detection_DL_Error~=0))/(Packet_Length*NT);

Detection_Orig_Error=S-Data_Orig_Mat;
BER_Orig=sum(sum(Detection_Orig_Error~=0))/(Packet_Length*NT);

% H_original
% Hx_new
% H_Estimate_DL_Complex_Mat_3
% abs(H_original)
% abs(Hx_new)
% abs(H_Estimate_DL_Complex_Mat_3)
