function [f10] = Fourier10(f)
% Keep first 10 terms in Fourier expansion 

ft = fft(f);

ft(1) = 0;
for i = 12:118
    ft(i) = 0;
end

f10 = ifft(ft);
end

