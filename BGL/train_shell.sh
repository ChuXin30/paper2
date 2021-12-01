#python LogKeyModel_train.py >> train.txt
#python new_template2vec_cnn_lstm_train.py >> train.txt
#python LogKeyModel_train.py noise/noise_0/WITH_S_train noise/noise_0_deeplog >> train.txt
#python new_template2vec_cnn_lstm_train.py noise/noise_0/WITH_S_train noise/noise_0_our >> train.txt
#python new_template2vec_cnn_lstm_train.py noise/noise_0/WITH_S_train noise/our_d500 >> train.txt
#python new_template2vec_cnn_lstm_train.py noise/noise_0/WITH_S_train noise/our_d400 >> train.txt
#python new_template2vec_cnn_lstm_train.py noise/drain_random_with_level_20/train noise/our_d400
#python new_template2vec_cnn_lstm_train.py noise/drain_random_with_level_20/train our_model/model_1 128 2 '2 3 4' 40 400 'noise/event_vector_top4.txt' 128 >> train.txt
#i=0
#for filename in `ls -t our_model/model_1`;#
#do
#  if (( $i %8 == 0 ))
#  then
#    python new_template2vec_cnn_lstm_predict.py 300 our_model/model_1/$filename noise_8/noise_0/WITH_S_test_normal_simple noise_8/noise_0/WITH_S_test_abnormal_simple 128 2 '2 3 4' 40 400 'noise/event_vector_top4.txt' >> method_cmp.txt
#  fi
#i=$(($i+1))
#done
#
#python new_template2vec_cnn_lstm_train.py noise/drain_random_with_level_20/train our_model/model_2 128 2 '2 3 4' 10 400 'noise/event_vector_top4.txt' 128 >> train.txt
#i=0
#for filename in `ls -t our_model/model_2`;#
#do
#  if (( $i %6 == 0 ))
#  then
#    python new_template2vec_cnn_lstm_predict.py 300 our_model/model_2/$filename noise_8/noise_0/WITH_S_test_normal_simple noise_8/noise_0/WITH_S_test_abnormal_simple 128 2 '2 3 4' 10 400 'noise/event_vector_top4.txt' >> method_cmp.txt
#  fi
#i=$(($i+1))
#done
#
#python new_template2vec_cnn_lstm_train.py noise/drain_random_with_level_20/train our_model/model_3 64 2 '2 3 4' 150 400 'noise/event_vector_top4.txt' 128 >> train.txt
#i=0
#for filename in `ls -t our_model/model_3`;#
#do
#  if (( $i %6 == 0 ))
#  then
#    python new_template2vec_cnn_lstm_predict.py 300 our_model/model_3/$filename noise_8/noise_0/WITH_S_test_normal_simple noise_8/noise_0/WITH_S_test_abnormal_simple 64 2 '2 3 4' 150 400 'noise/event_vector_top4.txt' >> method_cmp.txt
#  fi
#i=$(($i+1))
#done
#
#python new_template2vec_cnn_lstm_train.py noise/drain_random_with_level_20/train our_model/model_4 32 1 '2 3 4' 60 400 'noise/event_vector_top4.txt' 128 >> train.txt
#i=0
#for filename in `ls -t our_model/model_4`;#
#do
#  if (( $i %6 == 0 ))
#  then
#    python new_template2vec_cnn_lstm_predict.py 300 our_model/model_4/$filename noise_8/noise_0/WITH_S_test_normal_simple noise_8/noise_0/WITH_S_test_abnormal_simple 32 1 '2 3 4' 60 400 'noise/event_vector_top4.txt' >> method_cmp.txt
#fi
#i=$(($i+1))
#done
#
#python new_template2vec_cnn_lstm_train.py noise/drain_random_with_level_20/train our_model/model_5 32 2 '1 2 3 4' 80 400 'noise/event_vector_top4.txt' 128 >> train.txt
#i=0
#for filename in `ls -t our_model/model_5`;#
#do
#  if (( $i %6 == 0 ))
#  then
#    python new_template2vec_cnn_lstm_predict.py 300 our_model/model_5/$filename noise_8/noise_0/WITH_S_test_normal_simple noise_8/noise_0/WITH_S_test_abnormal_simple 32 2 '1 2 3 4' 80 400 'noise/event_vector_top4.txt' >> method_cmp.txt
#  fi
#i=$(($i+1))
#done

#python new_template2vec_cnn_lstm_train.py noise/drain_random_with_level_20/train our_model/model_6 256 3 '2 3 4' 10 400 'noise/event_vector_top4.txt' 64 >> train.txt
#i=0
#for filename in `ls -t our_model/model_6`;#
#do
#  if (( $i %6 == 0 ))
#  then
#    python new_template2vec_cnn_lstm_predict.py 300 our_model/model_6/$filename noise_8/noise_0/WITH_S_test_normal_simple noise_8/noise_0/WITH_S_test_abnormal_simple 256 3 '2 3 4' 10 400 'noise/event_vector_top4.txt' >> method_cmp.txt
#  fi
#i=$(($i+1))
#done
#
##python new_template2vec_cnn_lstm_train.py noise/drain_random_with_level_20/train our_model/model_7 64 2 '3 4 5' 150 400 'noise/event_vector_top4.txt' 128 >> train.txt
#i=0
#for filename in `ls -t our_model/model_7`;#
#do
#  if (( $i %6 == 0 ))
#  then
#    python new_template2vec_cnn_lstm_predict.py 300 our_model/model_7/$filename noise_8/noise_0/WITH_S_test_normal_simple noise_8/noise_0/WITH_S_test_abnormal_simple 64 2 '3 4 5' 150 400 'noise/event_vector_top4.txt' >> method_cmp.txt
#  fi
#i=$(($i+1))
#done

#python new_template2vec_cnn_lstm_predict.py 300 our_model/model_3/1_0.0017369518748520856_34 noise_8/noise_0/WITH_S_test_normal noise_8/noise_0/WITH_S_test_abnormal 64 2 '2 3 4' 150 400 'noise/event_vector_top4.txt' >> method_cmp.txt
#for((i=250;i<=600;i++));
#do
#  if (( $i %25 == 0 ))
#then
#  python new_template2vec_cnn_lstm_predict.py $i our_model/model_3/1_0.0016974644059958221_42 noise_8/noise_0/WITH_S_test_normal_simple noise_8/noise_0/WITH_S_test_abnormal_simple 64 2 '2 3 4' 150 400 'noise/event_vector_top4.txt' >> method_cmp.txt
#fi
#done
#python new_template2vec_cnn_lstm_predict.py 425 our_model/model_3/1_0.0016974644059958221_42 noise_8/noise_0/WITH_S_test_normal noise_8/noise_0/WITH_S_test_abnormal 64 2 '2 3 4' 150 400 'noise/event_vector_top4.txt' >> method_cmp.txt
python new_template2vec_cnn_lstm_train.py noise/drain_random_with_level_20/train our_model/model_8 64 2 '2 3 4' 300 400 'noise/event_vector_top4.txt' 64 >> train.txt
i=0
for filename in `ls -t our_model/model_8`;#
do
  if (( $i %6 == 0 ))
  then
    python new_template2vec_cnn_lstm_predict.py 425 our_model/model_8/$filename noise_8/noise_0/WITH_S_test_normal_simple noise_8/noise_0/WITH_S_test_abnormal_simple 64 2 '2 3 4' 300 400 'noise/event_vector_top4.txt' >> method_cmp.txt
  fi
i=$(($i+1))
done

