from .download_dem import process_dem
from .download_glc import process_glc
from .download_radar import process_radar

def main():
    process_dem()
    process_glc()
    # process_radar()

if __name__ == "__main__":
    main()