import os
import streamlit as st
from nltk.tokenize import sent_tokenize
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# change page icon and more
st.set_page_config(
    page_title="Video Summarizer",
    page_icon="🤖",
    layout="wide",

)


def extract_audio(video_path, audio_output_path):
    """extract audio from video that uploaded"""
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_output_path)


def main():
    st.title("Video Summarizer")
    video_file = st.file_uploader("Upload your video:", type=["wav", "mp4"])
    if "texts" not in st.session_state or not video_file:
        st.session_state.texts = []
    if "summ_text" not in st.session_state or not video_file:
        st.session_state.summ_text = None
    proc = st.button("generate")
    if video_file:
        if proc:
            with st.spinner():
                # handle stt model
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

                model_id = "distil-whisper/distil-large-v2"

                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
                )
                model.to(device)

                processor = AutoProcessor.from_pretrained(model_id)

                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    max_new_tokens=128,
                    torch_dtype=torch_dtype,
                    device=device,
                )
                # llm for summarization
                summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
                video_name = video_file.name.split('.')[0]
                audio_path = f"audio_data/audio_{video_name}.wav"
                video_path = f"video_data/video_{video_name}.mp4"
                if not os.path.exists(audio_path) and not os.path.exists(video_path):
                    with open(video_path, "wb") as file:
                        file.write(video_file.read())
                    extract_audio(video_path, audio_path)
                audio = AudioSegment.from_wav(audio_path)
                # Duration of each segment in milliseconds
                segment_duration = 30 * 1000

                # Calculate the number of segments
                num_segments = len(audio) // segment_duration

                # Segment the audio into 30-second intervals
                segments = [audio[i * segment_duration: (i + 1) * segment_duration] for i in range(num_segments)]

                # Handle the last segment (which may be shorter than 30 seconds)
                if len(audio) % segment_duration:
                    last_segment = audio[num_segments * segment_duration:]

                    segments.append(last_segment)

                audio_name = audio_path.split(".")[0]
                if not st.session_state.texts:
                    for i, segment in enumerate(segments):
                        if not os.path.exists(f"{audio_name}_{i}.wav"):
                            segment.export(f"{audio_name}_{i}.wav", format="wav")
                        # passing segments of audio to stt
                        result = pipe(f"{audio_name}_{i}.wav",
                                      generate_kwargs={"language": "english"},
                                      )

                        st.session_state.texts.append(result['text'])
                text = " ".join(st.session_state.texts)
                # segment text to fit for summarization llm
                if not st.session_state.summ_text:
                    text_sent = sent_tokenize(text)
                    if len(text_sent) <= 10:
                        st.session_state.summ_text = summarizer(text,
                                                                do_sample=False)[0]['summary_text']
                    else:
                        i = 0
                        res_text = " "
                        for _ in range(len(text_sent) // 10):
                            text_summ = " ".join(text_sent[i * 10:(i + 1) * 10])
                            res_text += \
                                summarizer(text_summ,
                                           do_sample=False)[0]['summary_text']
                            i += 1
                        if len(text_sent) % 10:
                            final_text = " ".join(text_sent[(i + 1) * 10:])
                            res_text += \
                                summarizer(final_text,
                                           do_sample=False)[0]['summary_text']
                        st.session_state.summ_text = res_text
        if st.session_state.summ_text:
            indx = st.session_state.summ_text.find("I am a prisoner")
            if indx:
                st.text_area("summarized text:",
                             value=st.session_state.summ_text[:indx],
                             height=600)
            else:
                st.text_area("summarized text:",
                             value=st.session_state.summ_text,
                             height=600)


if __name__ == "__main__":
    main()
