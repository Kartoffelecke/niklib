import typer
from niklib import image_tools
from pathlib import Path
from typing import Annotated

app = typer.Typer()
@app.command()
def extract_channels(
        filenames: list[Path], 
        channels: Annotated[
            str, typer.Option(help="comma separated list of channels to be extracted")
        ],
        outdir: Annotated[str, typer.Option(help="output directory")],
    ):
    channels = channels.split(",")
    channels = [int(x.strip()) for x in channels]
    for filename in filenames[1:]:
        print(f"{filename=}")
        extracted_image = image_tools.extract_channels(filename, channels)
        image_tools.imwrite(outdir / Path(filename.stem+"_extracted.tif"),extracted_image)

        

#@app.command()
#def rescale_intensity(filenames, channel, method):
#    print("test")
#    print(image_tools.extract_channels(filenames[0], [0]))
