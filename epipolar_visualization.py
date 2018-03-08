import cv2
import numpy as np
import random
import math
def normalized_eight(srcp,dstp):
	src_avgx,src_avgy=0,0
	dst_avgx,dst_avgy=0,0
	for i in range(len(srcp)):
		src_avgx+=srcp[i][0]
		src_avgy+=srcp[i][1]
		dst_avgx+=dstp[i][0]
		dst_avgy+=dstp[i][1]
	src_avg=[src_avgx/len(srcp),src_avgy/len(srcp)]
	dst_avg=[dst_avgx/len(dstp),dst_avgy/len(dstp)]
	src_dist,dest_dist = 0,0
	for i in range(0,len(srcp)):
		src_dist += math.sqrt(((srcp[i][0]-src_avg[0])**2)+((srcp[i][1]-src_avg[1])**2))
		dest_dist += math.sqrt(((dstp[i][0]-dst_avg[0])**2)+((dstp[i][1]-dst_avg[1])**2))
	scale_src = (math.sqrt(2))/(src_dist/len(srcp))
	scale_dst = (math.sqrt(2))/(dest_dist/len(dstp))
	t_src = np.asarray([[scale_src, 0, -src_avg[0]*scale_src],[0,scale_src,-src_avg[1]*scale_src], [0,0,1]])
	t_dst = np.asarray([[scale_dst, 0, -dst_avg[0]*scale_dst],[0,scale_dst,-dst_avg[1]*scale_dst], [0,0,1]])
	for i in range(len(srcp)):
		q,w    = srcp[i]
		srcp_n = np.dot(t_src,[q,w,1])
		srcp[i]= [srcp_n[0]/srcp_n[2],srcp_n[1]/srcp_n[2]]

		x,y    = dstp[i]
		dstp_n = np.dot(t_dst,[x,y,1])
		dstp[i]= [dstp_n[0]/dstp_n[2],dstp_n[1]/dstp_n[2]]
	a = np.zeros(shape = (8,9))
	k = 8
	eights = random.sample(range(0,len(srcp)),8)
	counter = 0
	while counter<k:
		entry = [dstp[counter][0]*srcp[counter][0], dstp[counter][0]*srcp[counter][1], dstp[counter][0], dstp[counter][1]*srcp[counter][0], dstp[counter][1]*srcp[counter][1], dstp[counter][1],srcp[counter][0], srcp[counter][1],1]
		a[counter] = entry
		counter+=1
	u,s,vt = cv2.SVDecomp(a)
	f = vt[-1].reshape(3,3)
	unor_f = np.linalg.multi_dot([t_dst,f,t_src])
	#rank2
	uf,df, vtf = cv2.SVDecomp(f)
	df[2] = 0
	f_prime = np.linalg.multi_dot(uf,df,vtf)
	return f_prime
def ransac(srcp,dstp):
	n = 10000
	main_count = 0
	inlier_ratio = 0
	final_f = []
	while main_count<n:
		eights = random.sample(range(0,len(srcp)),8)
		rand_src = []
		rand_dst = []
		lines	 = np.zeros(shape=(len(srcp),3))
		for i in range(8):
			rand_src[i] = srcp[eights[i]]
			rand_dst[i] = dstp[eights[i]]
		f = normalized_eight(rand_src, rand_dst)
		for i in range(len(srcp)):
			lines[i] = np.linalg.dot(f,[srcp[i][0],srcp[i][1],1])
		inlier_count = 0
		for i in range(len(srcp)):
			if(abs(lines[i][0]*dstp[i][0]+lines[i][1]*dstp[i][1]+lines[i][2])/math.sqrt(lines[i][0]**2+lines[i][1]**2)):
				inlier_count+=1
		if(inlier_count*100/len(srcp)>inlier_ratio):
			inlier_ratio = inlier_count*100/len(srcp)
			final_f = f
		w_s = (inliers/len(srcp))**4
		n=np.floor(abs(np.log(0.01)/np.log(abs(1-w_s))))
		main_count+=1
	final_inliers = []
	##STEP 6DAN DEVAM ET...

if __name__ == '__main__':
	img1 = cv2.imread("horse_0.JPG")
	img2 = cv2.imread("horse_20.JPG")
	orb = cv2.ORB_create(nfeatures=5000)
	kp_src,des_src 	= orb.detectAndCompute(img1,None)
	kp_dst, des_dst = orb.detectAndCompute(img2,None)
	print("Keypoints found.")
	bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
	matches = bf.match(des_src,des_dst)
	dmatches = sorted(matches, key = lambda x:x.distance)
	print("Keypoints matched.")
	src_pts  = np.float32([kp_src[m.queryIdx].pt for m in dmatches])
	dst_pts  = np.float32([kp_dst[m.trainIdx].pt for m in dmatches])
	f = normalized_eight(src_pts, dst_pts)
