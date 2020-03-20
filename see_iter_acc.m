clear;close all;
cc_file=dir('./caffe*');
ff=fopen('INFO.txt20170530-140806.5612','r');

content={};

while feof(ff)==0
    content=[content;fgetl(ff)];
end
fclose(ff);
iter=[];
loss=[];
acc=[];
acc_t3=[];
for u=1: length(content)
    iter_start=strfind(content{u},'218] Iteration');
    iter_end=strfind(content{u},',');
    iter_temp=str2num(content{u}(iter_start+15:iter_end-1));
    iter=[iter;iter_temp];
    
    loss_start=strfind(content{u},'Train net output #0: loss/loss = ');
    loss_end=strfind(content{u},'(');
    loss_temp=str2num(content{u}(loss_start+32:loss_end-1));
    loss=[loss;loss_temp];
    
    acc_start=strfind(content{u},'Test net output #0: accuracy@1 = ');
    acc_end=strfind(content{u},',');
    acc_temp=str2num(content{u}(acc_start+32:end));
    acc=[acc;acc_temp];
    
%     acc_t3_start=strfind(content{u},'Test net output #1: accuracy_top_3 = ');
%     %acc_end=strfind(content{u},',');
%     acc_t3_temp=str2num(content{u}(acc_t3_start+38:end));
%     acc_t3=[acc_t3;acc_t3_temp];
end


iter=iter';
acc=acc';
% acc_t3=acc_t3';
loss=loss';

figure(1);
% plot(loss);title('loss'); hold on;
% plot(acc);title('top1-accuracy');
subplot(121),plot(loss);title('loss')
subplot(122),plot(acc);title('top1-accuracy')
% subplot(133),plot(acc_t3);title('top3-accuracy')

saveas(figure(1),'lose_01.jpg')