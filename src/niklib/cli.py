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
        image_tools.imwrite(outdir / Path(f"{filename.stem}_ch{','.join(map(str, channels))}.tif"),extracted_image)


@app.command()
def rescale_intensity(
        filenames: list[Path],
        ):
    image_tools.rescale_intensity(filenames)

@app.command()
def lif_to_tif(input:Path, output:Path):
    image_tools.lif_to_tif(input, output)

@app.command()
def napari_load_tif(filenames: list[Path]):
    """open tifs in one napari window with their µm scaling applied"""
    import napari
    viewer = None
    for filename in filenames:
        viewer = image_tools.napari_load_tif(filename, viewer=viewer)
    napari.run()


#@app.command()
#def rescale_intensity(filenames, channel, method):
#    print("test")
#    print(image_tools.extract_channels(filenames[0], [0]))
