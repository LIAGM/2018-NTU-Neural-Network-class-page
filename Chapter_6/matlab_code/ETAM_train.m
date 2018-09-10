
% a function use Error Tolerant Associative Memory method training
% parameter "data" is a matrix with input and real output
% return single layer weights "w"
function w = ETAM_train(ptn)

	[ptn_h, ptn_w] = size(ptn);
	
	alpha = 0.005;
	
% 	w = ones(dimension, dimension);
% 	
% 	for i=1:dimension
%         if pattern_num ~= 1
%             
%             w(i,:) = w(i, :)/norm(w(i,:));
%         else
%             w(i,:) = 2*rand(1,dimension)-1;
%             w(i,:) = w(i, :)/norm(w(i,:));
%         end
% 	end
	
    w = ptn*ptn';
    for i=1:ptn_h
        w(i,:) = w(i, :)/norm(w(i,:));
    end

	w = [w zeros(ptn_h, 1)];
	
	ptn_1 = [ptn; -ones(1, ptn_w)];
	
	for i=1:ptn_h
        Cp = find(ptn(i,:)==1);
        Cn = find(ptn(i,:)==-1);
        
        dist = w(i,:)*ptn_1;
        [dp indexp] = min(dist(Cp));
        [dn indexn] = max(dist(Cn));
        
        if isempty(Cp)
            w(i,end) = 100;
            continue;
        elseif isempty(Cn)
            w(i,end) = -100;
            continue;
        else
            pre_dm = -100;
            DONE = 0;
            while (dp-dn)/2>pre_dm | (~DONE)
                DONE = 1;
                w(i,end) = w(i,end)+(dp+dn)/2;
                %rotate the hyperplane to increase the distances.
                w1 = w(i,1:(end-1)); %record previous w in case of recovering.
                w(i,1:(end-1)) = w(i,1:(end-1)) + alpha* ( ptn(:,Cp(indexp))'*ptn(i,Cp(indexp)) + ptn(:,Cn(indexn))'*ptn(i,Cn(indexn)) );
                

                
                %normalize w
                w(i,1:(end-1)) = w(i,1:(end-1))/norm(w(i,1:(end-1)));
                pre_dm = (dp-dn)/2;
                
                
                % if next step sp!=p
                row_ptn = ptn(i,:);
                sum = w(i,:)*ptn_1;
                
                %sum(find(sum~=0)) = sign(sum(find(sum~=0)));
                %sum(find(sum==0)) = sign(row_ptn(find(sum==0)));
                sum = sign(sum);
                sum(find(sum==0)) = 1;
                
                if any(sum~=ptn(i,:))
                    DONE = 0;
                end
                
                dist = w(i,:)*ptn_1;
                [dp indexp] = min(dist(Cp));
                [dn indexn] = max(dist(Cn));
                
            end
            
            
            w(i,1:(end-1)) = w1;    % recovery
            
            
            
        end
        
	end
    
end