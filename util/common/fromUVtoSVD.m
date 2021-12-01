function [U, S, V] = fromUVtoSVD(Up, Vp)
% Return U, S, V, such that USV' = UpVp' 
% where U, V are orthogonal and S is diagonal

[Qu, Ru] = qr(Up, 0) ;
[Qv, Rv] = qr(Vp, 0) ;
[U, S, V] = svd(Ru*Rv') ;
U = Qu * U ;
V = Qv * V ;

end