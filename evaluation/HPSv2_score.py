import hpsv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="results",
    )
    parser.add_argument(
        "--hps_version",
        type=str,
        help="hps_version",
        default="v2.0",
    )
    
    parser.add_argument(
        "--image_path",
        type=str,
        nargs="?",
        help="dir to evaluate",
        default="results",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    hpsv2.evaluate(args.image_path, hps_version=args.hps_version)
    
if __name__ == "__main__":
    main()