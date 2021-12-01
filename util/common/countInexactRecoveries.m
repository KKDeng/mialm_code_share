function nOut = countInexactRecoveries(diff, ref)
% Count the number of inexact recoveries
nOut = sum(diff >= 1e-4 * abs(ref)) ;
end