function out = flatMatrix(A, I, J)
% Flatten the matrix A on mask (I, J) :
% out(i) = A(I(i), J(i))

out = zeros(size(I)) ;
for i = 1:length(I)
    out(i) = A(I(i),J(i)) ;
end

end