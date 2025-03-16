import sys, os, pdb
import numpy

class Solution:
	def findMedianSortedArrays(self, nums1, nums2) :
		if not nums1:			
			finarr = nums2
			median = finarr[len(finarr)//2] if len(finarr)%2!=0 else (finarr[len(finarr)//2 - 1]+finarr[len(finarr)//2])/2
			return median    		
		if not nums2:
			finarr = nums1
			median = finarr[len(finarr)//2] if len(finarr)%2!=0 else (finarr[len(finarr)//2]+finarr[(len(finarr)//2)-1])/2
			return median
		
		arr = self.pusharray(nums1+nums2)
		finarr = []
		for j in range(1,len(arr)):
			val = self.minpop(arr)
			finarr.append(val)			
		median = finarr[len(finarr)//2] if len(finarr)%2!=0 else (finarr[len(finarr)//2]+finarr[(len(finarr)//2)-1])/2
		return float(median)       
	
	def pusharray(self, nums):
		arr = [0]
		for i in nums:
			arr.append(i)
			self.floatup(arr, len(arr)-1)
		return arr
			
	def swap(self, arr, i , j):
		arr[i],arr[j] = arr[j],arr[i]
	
	def floatup(self, arr, index):
		parent = index//2
		
		if index<1:
			return
		elif arr[parent]>arr[index]:
			self.swap(arr, parent, index)			
			self.floatup(arr, parent)

	def minpop(self, arr):
		if len(arr)==1:
			return False
		else:            
			self.swap(arr, 1, len(arr)-1) 
			minval = arr.pop()
			self.floatdown(arr, 1)            
			return minval
		
	def floatdown(self, arr, index):
		left = index*2 
		right = index*2 + 1
		smallest = index
		if len(arr)>left and arr[left]<arr[index]:
			smallest = left
		if len(arr)>right and arr[right]<arr[smallest]:
			smallest = right
		if smallest!=index:
			self.swap(arr, smallest, index)
			self.floatdown(arr, smallest)


		
		


arr1 = [2,3,4]
arr2 = [1]

obje = Solution()
median = obje.findMedianSortedArrays(arr1, arr2)
print(median)