#!/bin/bash

# Navigate to the root directory of the project.
cd "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"

# ================================= Edit Here ================================ #

BASE_IMAGE=osrf/ros
BASE_TAG=jazzy-desktop
ROS_NUMBER=2
IMAGE_NAME=control_arm_temperature

# =============================== Preliminaries ============================== #

mkdir -p build log src
if [[ $ROS_NUMBER == 1 ]]; then
    mkdir -p devel
elif [[ $ROS_NUMBER == 2 ]]; then
    mkdir -p install
fi

mkdir -p ~/.vscode ~/.vscode-server ~/.config/Code
mkdir -p ~/docker/${IMAGE_NAME}/Plotjuggler
cp docker/.config/PlotJuggler-3.ini ~/docker/${IMAGE_NAME}/Plotjuggler/PlotJuggler-3.ini

# =============================== Help Function ============================== #

helpFunction()
{
    echo ""
    echo "Usage: $0 [-l] [-r]"
    echo -e "\t-a   --all           Install all dependencies."
    echo -e "\t-d   --docs          Install sphynx."
    echo -e "\t-h   --help          Print the help."
    echo -e "\t-l   --larex         Install latex. Bigger, but can use LaTeX in plots."
    echo -e "\t-r   --rebuild       Rebuild the image."
    exit 1 # Exit script after printing help
}

# =============================== Build Options ============================== #

# Initialie the build options
DOCS=0
LATEX=0
REBUILD=0

# Auxiliary functions
die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }
no_arg() { if [ -n "$OPTARG" ]; then die "No arg allowed for --$OPT option"; fi; }

# Get the script options. This accepts both single dash (e.g. -a) and double dash options (e.g. --all)
while getopts adhlr-: OPT; do
    # support long options: https://stackoverflow.com/a/28466267/519360
    if [ "$OPT" = "-" ]; then     # long option: reformulate OPT and OPTARG
        OPT="${OPTARG%%=*}"       # extract long option name
        OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
        OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
    fi
    case "$OPT" in
        a | all )       no_arg; DOCS=1; LATEX=1 ;;
        d | docs )      no_arg; DOCS=1 ;;
        h | help )      no_arg; helpFunction ;;
        l | latex )     no_arg; LATEX=1 ;;
        r | rebuild )   no_arg; REBUILD=1 ;;
        ??* )           die "Illegal option --$OPT" ;;  # bad long option
        ? )             exit 2 ;;  # bad short option (error reported via getopts)
    esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

# ========================= Pull And Build The Image ========================= #

docker pull $BASE_IMAGE:$BASE_TAG

GID="$(id -g $USER)"

if [ "$REBUILD" -eq 1 ]; then
    cache="--no-cache"
else
    cache=""
fi

docker build \
    ${cache} \
    --build-arg BASE_IMAGE=$BASE_IMAGE \
    --build-arg BASE_TAG=$BASE_TAG \
    --build-arg ROS_NUMBER=$ROS_NUMBER \
    --build-arg MYUID=${UID} \
    --build-arg MYGID=${GID} \
    --build-arg USER=${USER} \
    --build-arg "PWDR=$PWD" \
    --build-arg DOCS=$DOCS \
    --build-arg LATEX=$LATEX \
    -t $IMAGE_NAME \
    -f docker/Dockerfile .
