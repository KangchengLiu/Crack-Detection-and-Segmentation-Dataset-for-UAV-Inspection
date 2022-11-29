import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="ALL", help='Which model weights to download')
args = parser.parse_args()

###############################
# VGG Net
###############################
# subprocess.check_output(['wget','http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'vgg_16_2016_08_28.tar.gz'])
# subprocess.check_output(['mv', 'vgg_16.ckpt', 'models'])
# subprocess.check_output(['rm', 'vgg_16_2016_08_28.tar.gz'])

# subprocess.check_output(['wget','http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'vgg_19_2016_08_28.tar.gz'])
# subprocess.check_output(['mv', 'vgg_19.ckpt', 'models'])
# subprocess.check_output(['rm', 'vgg_19_2016_08_28.tar.gz'])

###############################
# Inception Net
###############################
# subprocess.check_output(['wget','http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'inception_v1_2016_08_28.tar.gz'])
# subprocess.check_output(['mv', 'inception_v1.ckpt', 'models'])
# subprocess.check_output(['rm', 'inception_v1_2016_08_28.tar.gz'])

# subprocess.check_output(['wget','http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'inception_v2_2016_08_28.tar.gz'])
# subprocess.check_output(['mv', 'inception_v2.ckpt', 'models'])
# subprocess.check_output(['rm', 'inception_v2_2016_08_28.tar.gz'])

# subprocess.check_output(['wget','http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'inception_v3_2016_08_28.tar.gz'])
# subprocess.check_output(['mv', 'inception_v3.ckpt', 'models'])
# subprocess.check_output(['rm', 'inception_v3_2016_08_28.tar.gz'])

# subprocess.check_output(['wget','http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'inception_v4_2016_09_09.tar.gz'])
# subprocess.check_output(['mv', 'inception_v4.ckpt', 'models'])
# subprocess.check_output(['rm', 'inception_v4_2016_09_09.tar.gz'])

# subprocess.check_output(['wget','http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'inception_resnet_v2_2016_08_30.tar.gz'])
# subprocess.check_output(['mv', 'inception_resnet_v2.ckpt', 'models'])
# subprocess.check_output(['rm', 'inception_resnet_v2_2016_08_30.tar.gz'])


###############################
# ResNet V1
###############################
# subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'resnet_v1_50_2016_08_28.tar.gz'])
# subprocess.check_output(['mv', 'resnet_v1_50.ckpt', 'models'])
# subprocess.check_output(['rm', 'resnet_v1_50_2016_08_28.tar.gz'])

# subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'resnet_v1_101_2016_08_28.tar.gz'])
# subprocess.check_output(['mv', 'resnet_v1_101.ckpt', 'models'])
# subprocess.check_output(['rm', 'resnet_v1_101_2016_08_28.tar.gz'])

# subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'resnet_v1_152_2016_08_28.tar.gz'])
# subprocess.check_output(['mv', 'resnet_v1_152.ckpt', 'models'])
# subprocess.check_output(['rm', 'resnet_v1_152_2016_08_28.tar.gz'])


###############################
# ResNet V2
###############################
if args.model == "Res50" or args.model == "ALL":
	subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz'])
	subprocess.check_output(['tar', '-xvf', 'resnet_v2_50_2017_04_14.tar.gz'])
	subprocess.check_output(['mv', 'resnet_v2_50.ckpt', 'models'])
	subprocess.check_output(['rm', 'resnet_v2_50_2017_04_14.tar.gz'])

if args.model == "Res101" or args.model == "ALL":
	subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz'])
	subprocess.check_output(['tar', '-xvf', 'resnet_v2_101_2017_04_14.tar.gz'])
	subprocess.check_output(['mv', 'resnet_v2_101.ckpt', 'models'])
	subprocess.check_output(['rm', 'resnet_v2_101_2017_04_14.tar.gz'])

if args.model == "Res152" or args.model == "ALL":
	subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz'])
	subprocess.check_output(['tar', '-xvf', 'resnet_v2_152_2017_04_14.tar.gz'])
	subprocess.check_output(['mv', 'resnet_v2_152.ckpt', 'models'])
	subprocess.check_output(['rm', 'resnet_v2_152_2017_04_14.tar.gz'])

subprocess.check_output(['rm', 'train.graph'])
subprocess.check_output(['rm', 'eval.graph'])