# Enhanced-CCTV-system
Welcome to our project which implements the data logging of objects,vehicles,people during a cctv camera footage recording session
Lets see the steps for implementing and installing our project in your pc

Our System Specs:
	Processor	: Intel I7-8700k
	RAM	 	: 16gb ddr4 memory stick
	Graphics card	: Geforce Rtx 2060
	Storage space   : 2tb hdd,256gb ssd
	Operating System: Executed on Windows 11

Minimum specs Recommended
	Processor	: i5 processor
	RAM	 	: minimum 16gb ram
	Graphics card	: Any graphics card is better than integerated graphics.
	Storage space 	: 4gb storage space will be good to run it.
	OS		: Windows 8/10/11

STEPS FOR INSTALLATION:
STEP 1 FOR WINDOWS USERS: if you are going to be installing on Windows then you can skip the first three steps an follow this video directly:
		https://youtu.be/WK_2bpWj35A
STEP 1 for UBUNTU users : if you are trying on linux check this link out:
		https://medium.com/geekculture/yolov4-darknet-installation-and-usage-on-your-system-windows-linux-8dec2cea6e81
		this shows you the complete step by step process to install cuda and cudnn along with opencv and darknet.

STEP 2 : copy the two zip files(darknet-master.zip and sourcecode.zip) from the cd and and extract them side by side as darknet-master and sourcecode in the same working directory named project.

STEP 3 : install pycharm community edition from their official website on your pc.

Step 4 : replace the darknet.py and darknet_video.py from the darknet installation folder from the first step with the files given in the darknet-master folder extracted from the zip

Step 5 : now open the darknet-master/darknet folder with replaced files as pycharm project.

Step 6 : form the leftside projects tab open/double-click darknet_video.py. 

Step 7 : in the second line of the code:
		sys.path.insert(1,'/home/kadhiravan/Downloads/Project/sourcecode')
	replace the location with your location of the source code folder.
Step 8 : run the darknet_video.py with your webcam turned on.

Step 9 : END!!! check for the out put that will be displayed in a new window, also check the timeline.csv file for the datalogging and pics folder for the pics for the number plates collection.
