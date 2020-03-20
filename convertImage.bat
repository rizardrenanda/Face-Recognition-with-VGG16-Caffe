SET GLOG_logtostderr=1
D:\Dependencies\caffe-windows-ms\Build\x64\Release\convert_imageset.exe --gray="true" --shuffle --resize_width=96 --resize_height=64 D:/Dependencies/caffe-windows-ms/examples/caffe_homework/image/ ./train.txt ./train_lmdb

D:\Dependencies\caffe-windows-ms\Build\x64\Release\convert_imageset.exe --gray="true" --shuffle --resize_width=96 --resize_height=64 D:/Dependencies/caffe-windows-ms/examples/caffe_homework/image/ ./val.txt ./val_lmdb
pause