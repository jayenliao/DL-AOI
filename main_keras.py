'''
Deep Learning - HW5: AOI (keras version)
Jay Liao (re6094028@gs.ncku.edu.tw)
'''

from aoi_keras.args import init_arguments
from aoi_keras.trainer import Trainer
import tensorflow as tf

def main(args):
    preprocess_hyper = {
        'rescale': 1/args.rescale,
        #'width_shift_range': args.width_shift_range,
        #'height_shift_range': args.height_shift_range,
        'horizontal_flip': args.horizontal_flip,
        'vertical_flip': args.vertical_flip,
        #'brightness_range': args.brightness_range,
        #'zoom_range': args.zoom_range
    } 
    load_args = {
        'dataPATH': args.dataPATH,
        'trainDATA': args.trainDATA,
        'label': args.label,
        'batch_size': args.batch_size,
        'resize': args.resize,
        'val_size': args.val_size,
        'test_size': args.test_size,
        'preprocess_hyper': preprocess_hyper,
        'seed': args.seed
    }
    tf.config.list_physical_devices('GPU')
    
    trainer = Trainer(
        model_name=args.model_name,
        savePATH=args.savePATH,
        load_args=load_args,
        epochs=args.epochs,
        l2=args.l2,
        dropout=args.dropout,
        activation=args.activation
    )

    trainer.train()
    trainer.evaluate()
    trainer.predict(args.trainDATA.replace('train', 'test'))
    trainer.plot_training('loss', args.plot_figsize)
    trainer.plot_training('accuracy', args.plot_figsize)
    print('')

if __name__ == '__main__':
    args = init_arguments().parse_args()
    main(args)
