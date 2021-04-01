import os
import argparse
from tqdm import tqdm
import glob
from c3d import *
from classifier import *
from utils.visualization_util import *


# parser = argparse.ArgumentParser(description='Deep Anomorly Detection Survillance Video')
# parser.add_argument('Path',metavar='path',type=str,help='add input video',default='input/Explosion008_x264.mp4')
# args = parser.parse_args()

def process():
    path = 'input/*'
    for filename in tqdm(glob.glob(path)):
        print("process :" +filename)
        run_demo(filename)

def run_demo(video_path):

    video_name = os.path.basename(video_path).split('.')[0]
    #video_path =
    print("video_path " + video_path)
    print("video_name" + video_name)
 
    # read video
    video_clips, num_frames = get_video_clips(video_path)

    # build models
    feature_extractor = c3d_feature_extractor()
    classifier_model = build_classifier_model()

    # extract features
    rgb_features = []
    for i, clip in enumerate(tqdm(video_clips)):
        clip = np.array(clip)
        if len(clip) < params.frame_count:
            continue
        
        clip = preprocess_input(clip)
        rgb_feature = feature_extractor.predict(clip)[0]
        rgb_features.append(rgb_feature)

        

    rgb_features = np.array(rgb_features)

    rgb_feature_bag = interpolate(rgb_features, params.features_per_bag)

    predictions = classifier_model.predict(rgb_feature_bag)

    predictions = np.array(predictions).squeeze()

    predictions = extrapolate(predictions, num_frames)

    save_path = os.path.join(cfg.output_folder, video_name + '.mp4')
    print("save_path " , save_path)
    # visualize predictions
    visualize_predictions(video_path, predictions, save_path)


if __name__ == '__main__':
    process()