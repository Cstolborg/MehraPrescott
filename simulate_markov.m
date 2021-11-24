function [chain,state] = simulate_markov_v2(x,P,pi0,T)

%  x     = values assigned to each state: State mapping / functional etc.
%  P     = transition matrix
%  pi0    = probability distribution over initial state
%  T     = number of periods to simulate

% initialize state container
N = length(x);
state = zeros(N,T+1); 

% generate first x value (time 0 NOT time 1)
state(:,1) = rando(pi0); 

for t = 1:T
  state(:,t+1) = rando(P(state(:,t) == 1,:));
end

chain = x'*state;    