import argparse
from pathlib import Path

from openai import DefaultHttpxClient, OpenAI


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--language", help="Language code, such as zh or en; omit to detect"
    )
    args = parser.parse_args()
    language = {} if args.language is None else {"language": args.language}
    with OpenAI(
        base_url=args.url,
        api_key="EMPTY",
        max_retries=0,
        http_client=DefaultHttpxClient(trust_env=False),
    ) as client:
        with args.audio.open("rb") as audio:
            result = client.audio.transcriptions.create(
                model="whisper", file=audio, temperature=0, **language
            )
    print(result.text)


if __name__ == "__main__":
    main()
