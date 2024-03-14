# DOVER-CLIP-based-data-filtering-pipeline

## Inspiration

Generative AI is the hottest thing in the AI/ML landscape for both domain experts and laypeople alike. However, training a "good" generative AI model will correspondingly require "good" input data. As the size of models and datasets grows ever larger, there is a need for methods that can minimize the usage of sub-optimal input data during model training, which will then improve model performance and minimize wasted resources. Our inspiration was generously provided to us by Tiktok, who have also provided an initial exploratory dataset (in the form of text captions and corresponding Youtube URLs).

## What it does
Our pipeline is based on the aforementioned .json file provided to us from Tiktok. The steps of the pipeline are as follows:

1. The .json file is parsed and saved as a .csv file for compatibility. Parameters including video URLs, clip timings, and text captions are saved.
2. A subset of the videos (due to processing, time and memory constraints) is extracted from the .csv dataset.
3. Initial filtering is done, such that videos with relatively low FPS or clips with abnormally low durations are removed.
4. The videos are downloaded from YouTube via their URLs, and processed into both .mp4 clips and keyframes.
5. For each clip: The average CLIP score is computed between the initial, middle and last frames, and the corresponding text caption, to assess the caption-video alignment of the clip.
6. The video's aesthetic and technical quality is assessed using DOVER, an open-source pre-trained video quality assessment model.
7. An overall score is computed from the previous three scores, and assigned to each clip.
8. Based on the overall score, a cutoff threshold can be used to select the top N caption-clip pairs.

## How we built it

Our pipeline is influenced by our (lack of) expertise, time and processing power constraints, and relative inexperience in this domain. Python-based Jupyter Notebook/Google Collab notebooks were used to standardize development and allow for relative ease of debugging. Open-source and pre-trained models are used in both the text-image alignment and video quality assessments, as our lack of specialized hardware and time makes training a new model infeasible. As the pipeline could only be tested on our local hardware/Google Collab, we chose to specifically use relatively light-weight models to minimize inference time.

## Explanatory YouTube URL

https://youtu.be/iRGKJqZFatM
