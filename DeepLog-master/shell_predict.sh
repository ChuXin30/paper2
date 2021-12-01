
#i=0
#for filename in `ls -t cnn_lstm_model_pramters/window_size_10\(1\)`;#
#do
#  if (( $i %10 == 0 ))
#then
    #echo $filename
    #python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters\/window_size_10\(1\)\/$filename >> 'cnn_lstm_model_pramters/window_size_10(1).txt'
#    python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters/window_size_10\(1\)/$filename 2 64 150 '2 3 4' 10 8 >> 'cnn_lstm_model_pramters/window_size_10(1).txt'
#fi
#i=$(($i+1))
#done

#layer3
#i=0
#for filename in `ls -t cnn_lstm_model_pramters/layer_3`;#
#do
#  if (( $i %10 == 0 && i <= 200  ))
#then
    #echo $filename
    #python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters\/window_size_10\(1\)\/$filename >> 'cnn_lstm_model_pramters/window_size_10(1).txt'
#    python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters/layer_3/$filename 3 64 150 '2 3 4' 10 8 >> 'cnn_lstm_model_pramters/layer_3.txt'
#fi
#i=$(($i+1))
#done

#layer4
#i=0
#for filename in `ls -t cnn_lstm_model_pramters/layer_4`;#
#do
#  if (( $i %10 == 0 && i <= 200  ))
#then
    #echo $filename
    #python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters\/window_size_10\(1\)\/$filename >> 'cnn_lstm_model_pramters/window_size_10(1).txt'
#    python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters/layer_4/$filename 4 64 150 '2 3 4' 10 8 >> 'cnn_lstm_model_pramters/layer_4.txt'
#fi
#3i=$(($i+1))
#done

#layer5
#i=0
#for filename in `ls -t cnn_lstm_model_pramters/layer_5`;#
#do
#  if (( $i %10 == 0 && i <= 200  ))
#then
    #echo $filename
    #python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters\/window_size_10\(1\)\/$filename >> 'cnn_lstm_model_pramters/window_size_10(1).txt'
#    python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters/layer_5/$filename 5 64 150 '2 3 4' 10 8 >> 'cnn_lstm_model_pramters/layer_5.txt'
#fi
#i=$(($i+1))
#done

#lstm_32
#i=0
#for filename in `ls -t cnn_lstm_model_pramters/lstm_32`;#
#do
#  if (( $i %10 == 0 && i <= 200  ))
#then
#    python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters/lstm_32/$filename 2 32 150 '2 3 4' 10 8 >> 'cnn_lstm_model_pramters/lstm_32.txt'
#fi
#i=$(($i+1))
#done

#lstm_128
#i=0
#for filename in `ls -t cnn_lstm_model_pramters/lstm_128`;#
#do
#  if (( $i %10 == 0 && i <= 200  ))
#then
#    python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters/lstm_128/$filename 2 128 150 '2 3 4' 10 8 >> 'cnn_lstm_model_pramters/lstm_128.txt'
#fi
#i=$(($i+1))
#done

#lstm_256
#i=0
#for filename in `ls -t cnn_lstm_model_pramters/lstm_256`;#
#do
#  if (( $i %10 == 0 && i <= 200  ))
#then
#    python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters/lstm_256/$filename 2 256 150 '2 3 4' 10 8 >> 'cnn_lstm_model_pramters/lstm_256.txt'
#fi
#i=$(($i+1))
#done

#kernel_50
#i=0
#for filename in `ls -t cnn_lstm_model_pramters/kernel_50`;#
#do
#  if (( $i %10 == 0 && i <= 200  ))
#then
    #echo $filename
    #python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters\/window_size_10\(1\)\/$filename >> 'cnn_lstm_model_pramters/window_size_10(1).txt'
#    python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters/kernel_50/$filename 2 64 50 '2 3 4' 10 8 >> 'cnn_lstm_model_pramters/kernel_50.txt'
#fi
#i=$(($i+1))
#done

#kernel_100
#i=0
#for filename in `ls -t cnn_lstm_model_pramters/kernel_100`;#
#do
#  if (( $i %10 == 0 && i <= 200 ))
#then
    #echo $filename
    #python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters\/window_size_10\(1\)\/$filename >> 'cnn_lstm_model_pramters/window_size_10(1).txt'
#    python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters/kernel_100/$filename 2 64 100 '2 3 4' 10 8 >> 'cnn_lstm_model_pramters/kernel_100.txt'
#fi
#i=$(($i+1))
#done

#kernel_200
#i=0
#for filename in `ls -t cnn_lstm_model_pramters/kernel_200`;#
#do
#  if (( $i %10 == 0 && i <= 200  ))
#then
#    python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters/kernel_200/$filename 2 64 200 '2 3 4' 10 8 >> 'cnn_lstm_model_pramters/kernel_200.txt'
#fi
#i=$(($i+1))
#done

#kernel_2_3
i=0
for filename in `ls -t cnn_lstm_model_pramters/kernel_2_3`;#
do
  if (( $i %10 == 0 && i <= 200  ))
then
    python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters/kernel_2_3/$filename 2 64 150 '2 3' 10 8 >> 'cnn_lstm_model_pramters/kernel_2_3.txt'
fi
i=$(($i+1))
done

#kernel_3_4
i=0
for filename in `ls -t cnn_lstm_model_pramters/kernel_3_4`;#
do
  if (( $i %10 == 0 && i <= 200  ))
then
    python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters/kernel_3_4/$filename 2 64 150 '3 4' 10 8 >> 'cnn_lstm_model_pramters/kernel_3_4.txt'
fi
i=$(($i+1))
done

#kernel_3_4_5
i=0
for filename in `ls -t cnn_lstm_model_pramters/kernel_3_4_5`;#
do
  if (( $i %10 == 0 && i <= 200  ))
then
    python new_template2vec_cnn_lstm_predict.py cnn_lstm_model_pramters/kernel_3_4_5/$filename 2 64 150 '3 4 5' 10 8 >> 'cnn_lstm_model_pramters/kernel_3_4_5.txt'
fi
i=$(($i+1))
done