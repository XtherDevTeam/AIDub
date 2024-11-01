# AIDub - The All-in-One AI Dubbing Solution to SAG-AFTRA 2024 Video Game Strike

## Abstract

In 2024, SAG-AFTRA announced a video game strike against the abuse of AI technology. The strike has been going on for a long time, and Hoyoverse Games which were not related to the reason of the strike has also been widely affected. As a plotline enjoyer and an English learner, I can not bear playing games without voice overs. Therefore, Here comes the all-in-one AI dubbing solution for the current situation.

The original purpose of this project is to help those people who suffered the same pain as mine. As a citizen of People's Republic of China, a Socialist, I fully understand the purpose and the background of the strike and support whatever the union done for the labors. Nevertheless, in my opinion of view, fears of AI technology advance are just the fears of `being replaced`.

To be more precise, it is the fear of the capitalists will use AI technology to replace the human labor, and maximize their profit, which is a representation of the basic contradiction of capitalism, the contradiction between socialization of production and private appropriation of the means of production, in Marxist terms.

As far as I am concerned, I sincerely wish that there will be one day, everyone can gain the equal developing capability and embrace the benefits of AI technology advance.

# Theories and Methods

As I mentioned in the video, as a skillful 15 yo Python programmer, it would be a shame for me if I can't implement this project.

Basically, by combining the websprider, AI technology, OCR, I created an all-in-one AI dubbing solution for the current situation due to SAG-AFTRA 2024 Video Game strike.

## Technologies Used

For gathering the dataset, and missing subtitles, I used the `beautifulsoup4` library to fetch the subtitles and voiceovers from the pages declared in `sources_to_fetch` and `source_text_to_dub` of `config.py` file.

For the voice changer and AI dubbing, my final selection is `GPT-SoVITS` which is a all-in-one TTS solution that included a variety of features like fine-tuning model, and inference API server. It is a powerful library to create high-quality AI voice overs which I have been using it for my projects for a long time.

For the final Detect and Play (DnP) feature, I used `PaddleOCR` to detect the text in bottom half screenshots and used `pydub` to play the voice overs.

## The process

1. Gathering the dataset:
    - Program will first find the certain traits of target pages declared in `sources_to_fetch` and `source_text_to_dub` of `config.py` file, extracting the subtitles and voiceovers.
    - Then they will be saved in the corresponding paths.
    - The datasets used to train the GPT-SoVITS model will be automatically transformed into usable format for `GPT-SoVITS` library.
    - The subtitles need to be dub will be saved in the `dub_manifest.json`.
2. Fine-tuning the GPT-SoVITS model:
    - Next, user should use GPT-SoVITs on Google Colab or any other platform (Except for WSL, due to unknown errors) to fine-tune the model on their own dataset through `WebUI` of `GPT-SoVITS`
    - Finally, user should store the models (`.ckpt` and `.pth` files) in the `thirdparty/GPTSoViTs/GPT_weights_v2` and `thirdparty/GPTSoViTs/SoVITS_weights` directories correspondingly of the project.
3. Emotion classification:
    - The `transformer` library is used to classify the emotions of the voice overs.
    - After fetching the dataset, by running `app.py` file with `--emotion-classification` argument, the program will start the emotion classification process.
    - It will traverse the `dataset_manifest.json` file and classify the emotions of the voice overs into `anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust` categories.
    - The emotion classification results will be saved in the `sentiment_analysis.json` file.
3. Dubbing the subtitles:
    - By starting `app.py` file with `--inference-server` argument, the program will start the inference server of GPT-SoVITS.
    - Then, start the `app.py` file with `--dub-all` argument to start the dubbing process.
    - The program will find the subtitles declared in `dub_manifest.json` file and dub them using the GPT-SoVITS model.
    - During the process, there are two methods for selection reference voice overs:
        - `pick_random`: randomly select a voice over from the dataset.
        - `pick_by_emotion`: select the closest voice over to the target voice over based on the emotion classification results.
    - The dubbed subtitles will be saved in the `dub_result` directory, with the result manifest file stored in `dub_result_manifest.json` used for the DnP feature.
4. Detect and Play (DnP):
    - By starting `dnp.py` file, Detect and Play (DnP) feature will be activated.
    - The program will constantly monitor the screenshots to detect the text in the bottom half of the screen.
    - If the text is detected, the program will play the corresponding voice over using `pydub` library.