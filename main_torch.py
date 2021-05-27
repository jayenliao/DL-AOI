'''
Deep Learning - HW5: AOI (torch version)
Jay Liao (re6094028@gs.ncku.edu.tw)
'''

from aoi_torch.trainer import Trainer
from aoi_torch.args import init_arguments
from aoi_torch.load_data import load_with_splitting

def main(args):
    # load and preprocess the data
    transforms_hyper = {
        'pretrained_size': args.pretrained_size,
        'padding': args.padding,
        'rotation': args.rotation,
        'horizontal_flip': args.horizontal_flip
    }
    data_tr, data_va, data_te, data_test = load_with_splitting(args.label, args.val_size, args.test_size, args.seed, transforms_hyper)

    # put the data into the trainer
    trainer = Trainer(
        data_tr, data_va, data_te,
        args.device, args.pretrained_model,
        args.optimizer, args.lr, args.epochs, args.batch_size,
        args.savePATH, args.verbose, args.seed
    )

    trainer.train(args.print_result_per_epochs)  # train and save the model
    trainer.evaluate()                           # evaluate the model on the splitting testing data
    trainer.predict(args.testDATA, args.label, data_test) # predict for the real testing data without ground truth labels
    trainer.plot_training('loss', args.figsize, save_plot=True)
    trainer.plot_training('accuracy', args.figsize, save_plot=True)

if __name__ == '__main__':
    args = init_arguments().parse_args()
    main(args)
