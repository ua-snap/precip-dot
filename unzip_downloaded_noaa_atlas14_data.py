# unzip the data in parallel

def unzip( fn ):
	print(fn)
	command = 'unzip {}'.format( fn ) 
	return os.system( command )

if __name__ == '__main__':
	import multiprocessing as mp
	import os, glob

	# list the data
	files = glob.glob('/workspace/Shared/Tech_Projects/DOT/project_data/NOAA_Atlas14/raw/zip/*.zip')

	os.chdir('/workspace/Shared/Tech_Projects/DOT/project_data/NOAA_Atlas14/raw/extracted')
	pool = mp.Pool( 32 )
	pool.map( unzip, files )
	out = pool.close()
	pool.join()