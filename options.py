import argparse

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Active-DA-Classification')

        ''' Runs '''
        parser.add_argument('--desc', type=str, default="default",
                            help='description for this experiment')
        parser.add_argument('--session', type=str, default="default")

        ''' Dataset '''
        parser.add_argument('--dataset', type=str, default='office_home',
                            choices=['office_home', 'office_home_rsut', 'domainnet', 'visda17'],
                            help='the name of dataset')
        parser.add_argument('--source', type=str, default='Art',
                            help='source domain')
        parser.add_argument('--target', type=str, default='Clipart',
                            help='target domain')
        parser.add_argument('--budget', type=float, default=0.1,
                            help='budget for active learning')

        ''' running mode '''
        parser.add_argument('--resume', type=str, default=None, help='path to pth')
        parser.add_argument('--resume_training', action='store_true', default=False)
        parser.add_argument('--testonly', action='store_true', default=False)

        ''' model '''
        parser.add_argument('--method', type=str, default='DANN_ESTIMATED_SEMI_PMMD_ONLINE', help='method')
        parser.add_argument('--net', type=str, default='resnet50',
                            choices=['resnet50'], help='which network to use')
        parser.add_argument('--scratch', default=False, action='store_true')

        ''' optimization '''
        parser.add_argument('--bs', type=int, default=20)
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: 0.001)')
        parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay (default: 0.0005)')
        parser.add_argument('--max_epoch', type=int, default=100)
        parser.add_argument('--start_step', type=int, default=-1)

        ''' resource options '''
        parser.add_argument('--num_workers', type=int, default=9)
        parser.add_argument('--unl_num_workers', type=int, default=9)

        ''' logging '''
        parser.add_argument('--dontlog', action='store_true', default=False,
                            help='control wandb logging (Not logging)')
        parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging '
                                'training status')
        parser.add_argument('--save_interval', type=int, default=100, metavar='N',
                            help='how many batches to wait before saving a model')

        ''' misc '''
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        self.parser = parser

    def modify_command_options(self, args):
        if(args.dataset == 'office_home' or args.dataset == 'office_home_rsut'):
            args.ncls = 65
        elif(args.dataset == 'domainnet'):
            args.ncls = 345
        elif(args.dataset == 'visda17'):
            args.ncls = 12
        else:
            raise NotImplementedError

        ''' Modify session '''
        if args.dataset == 'domainnet':
            data_code = 'D'
        elif args.dataset == 'office_home':
            data_code = 'O'
        elif args.dataset == 'visda17':
            data_code = 'V'
        elif args.dataset == 'office_home_rsut':
            data_code = 'OU'
        else:
            raise NotImplementedError
        args.session = args.session + '-b{}_{}{}{}2{}'.format(
                                                args.budget,
                                                data_code,
                                                args.net.capitalize()[0],
                                                args.source.capitalize()[0],
                                                args.target.capitalize()[0]
                                                )

        ''' Logging '''
        args.wandb_log = not args.dontlog

        return args

    def parse(self):
        args = self.parser.parse_args()
        return args
