#for((i=0;i<=1900;i++));
#do
#  if (( $i %50 == 0 ))
#then
#    python new_template2vec_cnn_lstm_predict.py $i >> noise_2/our_predict.txt
#fi
#done
#i=0
#for filename in `ls -t noise/our_model_random`;#
#do
#  if (( $i %8 == 0 ))
#then
#    #echo $filename
#    python new_template2vec_cnn_lstm_predict.py 300 noise/our_model_random/$filename noise_2/noise_0/WITH_S_test_normal_simple noise_2/noise_0/WITH_S_test_abnormal_simple >> noise_2/our_predict.txt
#fi
#i=$(($i+1))
#done
#python new_template2vec_cnn_lstm_predict.py 300 noise/our_model_random/1_0.0022038982871419015_14 noise_3/noise_0/WITH_S_test_normal_simple noise_3/noise_0/WITH_S_test_abnormal_simple >> method_cmp.txt
#python new_template2vec_cnn_lstm_predict.py 300 noise/our_model_random/1_0.0022038982871419015_14 noise_4/noise_0/WITH_S_test_normal_simple noise_4/noise_0/WITH_S_test_abnormal_simple >> method_cmp.txt
#python new_template2vec_cnn_lstm_predict.py 300 noise/our_model_random/1_0.0022038982871419015_14 noise_5/noise_0/WITH_S_test_normal_simple noise_5/noise_0/WITH_S_test_abnormal_simple >> method_cmp.txt
#python new_template2vec_cnn_lstm_predict.py 300 noise/our_model_random/1_0.0022038982871419015_14 noise_2/noise_0/WITH_S_test_normal_simple noise_2/noise_0/WITH_S_test_abnormal_simple >> method_cmp.txt
#python LogKeyModel_predict.py 200 noise_2/noise_0/Deeplog_test_normal_simple noise_2/noise_0/Deeplog_test_abnormal_simple >> method_cmp.txt
#python LogKeyModel_predict.py 200 noise_3/noise_0/Deeplog_test_normal_simple noise_3/noise_0/Deeplog_test_abnormal_simple >> method_cmp.txt
#python LogKeyModel_predict.py 200 noise_4/noise_0/Deeplog_test_normal_simple noise_4/noise_0/Deeplog_test_abnormal_simple >> method_cmp.txt
#python LogKeyModel_predict.py 200 noise_5/noise_0/Deeplog_test_normal_simple noise_5/noise_0/Deeplog_test_abnormal_simple >> method_cmp.txt
#python LogKeyModel_predict.py 200 noise_2/noise_0/WITH_S_test_normal_simple noise_2/noise_0/WITH_S_test_abnormal_simple >> method_cmp.txt
#python LogKeyModel_predict.py 200 noise_3/noise_0/WITH_S_test_normal_simple noise_3/noise_0/WITH_S_test_abnormal_simple >> method_cmp.txt
#python LogKeyModel_predict.py 200 noise_4/noise_0/WITH_S_test_normal_simple noise_4/noise_0/WITH_S_test_abnormal_simple >> method_cmp.txt
#python LogKeyModel_predict.py 200 noise_5/noise_0/WITH_S_test_normal_simple noise_5/noise_0/WITH_S_test_abnormal_simple >> method_cmp.txt
#i=0
#for filename in `ls -t noise/our_d500`;#
#do
#  if (( $i %2 == 0 ))
#then
#    #echo $filename
#    python new_template2vec_cnn_lstm_predict.py 300 noise/our_d500/$filename noise_8/noise_0/WITH_S_test_normal_simple noise_8/noise_0/WITH_S_test_abnormal_simple >> method_cmp.txt
#fi
#i=$(($i+1))
#done
#i=0
#for filename in `ls -t noise/our_d400`;#
#do
#  if (( $i %4 == 0 ))
#then
#    #echo $filename
#    python new_template2vec_cnn_lstm_predict.py 300 noise/our_d400/$filename noise_8/noise_0/WITH_S_test_normal_simple noise_8/noise_0/WITH_S_test_abnormal_simple >> method_cmp.txt
#fi
#i=$(($i+1))
#done
#for((i=1;i<=600;i++));
#do
#  if (( $i %50 == 0 ))
#then
#    python new_template2vec_cnn_lstm_predict.py $i noise/our_d400/1_0.0018131568025345015_29 noise_8/noise_0/WITH_S_test_normal_simple noise_8/noise_0/WITH_S_test_abnormal_simple >> method_cmp.txt
#fi
#done
#python new_template2vec_cnn_lstm_predict.py 300 noise/our_1/1_0.002024907733326758_22  noise_8/noise_0/WITH_S_test_normal_simple noise_8/noise_0/WITH_S_test_abnormal_simple >> method_cmp.txt
#i=0
#for filename in `ls -t noise/our_d400`;#
#do
#  if (( $i %4 == 0 ))
#  then
#    python new_template2vec_cnn_lstm_predict.py 300 noise/our_d400/$filename noise_8/noise_0/WITH_S_test_normal_simple noise_8/noise_0/WITH_S_test_abnormal_simple 64 2 '2 3 4' 100 400 'noise/event_vector_top4.txt' >> method_cmp.txt
#fi
#done
