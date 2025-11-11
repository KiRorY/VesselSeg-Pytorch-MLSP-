from config import parse_args
from train import main as train_main
from test import main as test_main


def main():
    """Main entry point"""
    args = parse_args()
    
    print('\n' + '='*60)
    print('Vessel Segmentation - Unified Pipeline')
    print('='*60)
    print(f'Mode: {args.mode}')
    print(f'Experiment: {args.save}')
    print('='*60)
    
    if args.mode == 'train':
        print('\n[TRAINING MODE]')
        train_main()
    
    elif args.mode == 'test':
        print('\n[TESTING MODE]')
        test_main()
    
    elif args.mode == 'full':
        print('\n[FULL PIPELINE: TRAIN + TEST]')
        print('\n--- Step 1: Training ---')
        train_main()
        print('\n--- Step 2: Testing ---')
        test_main()
    
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'train', 'test', or 'full'")
    
    print('\n' + '='*60)
    print('ALL TASKS COMPLETED!')
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
