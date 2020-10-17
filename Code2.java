package test;

import java.security.AllPermission;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.NavigableSet;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.SortedMap;
import java.util.Stack;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.DelayQueue;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.jws.Oneway;
import javax.naming.spi.DirStateFactory.Result;
import javax.swing.event.ListSelectionEvent;
import javax.swing.text.Segment;

public class test {
	
	public static boolean isNavigable(final String src, final String dst, final Set<String> dictionary) {
	    if (src.length() != dst.length()) {
	        return false;
	    }
	    if (src.equals(dst)) {
	        return true;
	
	    }
	    dictionary.remove(src);
	
	    final Queue<String> q = new ArrayDeque<String>();
	    q.add(src);
	
	    while (!q.isEmpty()) {
	        final String intermediate = q.poll();
	
	        for (int i = 0; i < intermediate.length(); i++) {
	        	char[] candidateChars = intermediate.toCharArray();
	            for (char j = 'a'; j < 'z'; j++) {
	                candidateChars[i] = j;
	
	                final String candidate = new String(candidateChars);
	
	                if (candidate.equals(dst)) {
	                    System.out.print("-->" + candidate);
	                    return true;
	                } else if (dictionary.contains(candidate)) {
	                    dictionary.remove(candidate);
	                    q.add(candidate);
	                    System.out.print("-->" + candidate);
	                }
	            }
	        }
	        
	        System.out.println();
	    }
	
	    return false;
	}
	
	public static List<List<String>> wordLadderAll(Set<String> dictionary, String src, String dst){
		if(src == null || dst == null || dictionary == null || src.isEmpty() || dst.isEmpty() || dictionary.isEmpty()){
			return Collections.EMPTY_LIST;
		}
		//Queue to traverse in BFS
		Queue<String> queue = new ArrayDeque<String>();
		//path from a node to its parent along the BFS traversal
		Map<String, String> parent = new HashMap<String, String>();
		//level or length of a word appeared in the DAG
		Map<String, Integer> pathLen = new HashMap<String, Integer>();
		//min length path so far
		int minLen = Integer.MAX_VALUE;
		//resulting shortest paths
		List<List<String>> paths =  new ArrayList<>();
		//resulting shortest path last nodes
		Set<String> shortestPathLeaves = new HashSet<String>();
		
		//add source to queue to start traversing
		queue.add(src);
		pathLen.put(src, 0);
		while(!queue.isEmpty()){
			String intermediate = queue.poll();
			//we already have a shortest path, so discard this longer path
			if(pathLen.get(intermediate) >= minLen){
				continue;
			}
			
			//BFS to each possible 1 edit distance neighbors in dictionary
			for(int i = 0; i<intermediate.length(); i++){
				char[] candidateChars = intermediate.toCharArray();
				//all possible words with current character variations
				for(char c = 'a'; c < 'z'; c++){
					candidateChars[i] = c;
					String candidate = new String(candidateChars);
					
					if(!pathLen.containsKey(candidate)){
						pathLen.put(candidate, Integer.MAX_VALUE);
					}
					//Dijktra's shortest path formullae
					if(pathLen.get(intermediate)+1 > pathLen.get(candidate)){
						continue;
					}
					
					//if we reach a solution, add it to solution
					if(candidate.equals(dst)){
						shortestPathLeaves.add(intermediate);
						minLen = Math.min(minLen, pathLen.get(intermediate)+1);
					}
					//otherwise if this intermediate word is present in dictionary then 
					//add it as children and update the path len
					else if(dictionary.contains(candidate)){
						parent.put(candidate, intermediate);
						pathLen.put(candidate, pathLen.get(intermediate)+1);
						queue.add(candidate);
					}
				}
			}
		}
		
		//Add all paths to result set
		for(String path : shortestPathLeaves){
			paths.add(getPath(parent, path, src, dst));
		}
		
		return paths;
	}
	
	private static List<String> getPath(Map<String, String> parentMap, String leaf, String src, String dst){
		List<String> path = new ArrayList<String>();
		
		String node = leaf;
		path.add(dst);
		path.add(0, leaf);
		while(parentMap.get(node) != null && parentMap.get(node) != src){
			node = parentMap.get(node);
			path.add(0, node);
		}
		path.add(0, src);
		
		return path;
	}
	
	public static BTNode rightMostCousin(BTNode root, int targetKey){
		LinkedList<BTNode> q = new LinkedList<BTNode>();
		
		int count = 0;
		q.add(root);	
		count++;
		boolean targetLevel = false;
		
		while(!q.isEmpty())
		{
			BTNode node = q.remove();	
			count--;
			if((node.left!=null && node.left.key == targetKey) || (node.right!=null && node.right.key == targetKey))
				targetLevel = true;			
			
			if(node.left != null) q.add(node.left);
			if(node.right != null) q.add(node.right);
	
			if(count == 0){			
				count = q.size();
				if(targetLevel){
					BTNode cousin = null;
					while(!q.isEmpty()){
						cousin = q.remove();															
					}
									
					return cousin;
				}
			}
		}
		
		return null;
	}
	
	public static BTNode rightMostCousin2(BTNode root, int targetKey){
		LinkedList<BTNode> q = new LinkedList<BTNode>();
		
		int count = 0;
		q.add(root);	
		count++;
		boolean targetLevel = false;
		
		while(!q.isEmpty())
		{
			BTNode node = q.remove();	
			count--;
			if(node.key == targetKey)
				targetLevel = true;			
			
			if(node.left != null) q.add(node.left);
			if(node.right != null) q.add(node.right);
	
			if(count == 0){			
				count = q.size();
				if(targetLevel){
					if(node.key != targetKey)
						return node;
					else return null;
				}
			}
		}
		
		return null;
	}
	
	public static int size(final TreeNode node) {
	    return node == null ? 0 : node.size;
	}

	public static int height(final TreeNode node) {
	    return node == null ? 0 : node.height;
	}

	public static TreeNode rotateLeft(final TreeNode root) {
	    final TreeNode newRoot = root.right;
	    final TreeNode leftSubTree = newRoot.left;

	    newRoot.left = root;
	    root.right = leftSubTree;

	    root.height = max(height(root.left), height(root.right)) + 1;
	    newRoot.height = max(height(newRoot.left), height(newRoot.right)) + 1;

	    newRoot.size = size(newRoot.left) + size(newRoot.right) + 1;
	    newRoot.size = size(newRoot.left) + size(newRoot.right) + 1;

	    return newRoot;
	}

	public static TreeNode rotateRight(final TreeNode root) {
	    final TreeNode newRoot = root.left;
	    final TreeNode rightSubTree = newRoot.right;

	    newRoot.right = root;
	    root.left = rightSubTree;

	    root.height = max(height(root.left), height(root.right)) + 1;
	    newRoot.height = max(height(newRoot.left), height(newRoot.right)) + 1;

	    newRoot.size = size(newRoot.left) + size(newRoot.right) + 1;
	    newRoot.size = size(newRoot.left) + size(newRoot.right) + 1;

	    return newRoot;
	}

	public static int max(final int a, final int b) {
	    return a >= b ? a : b;
	}

	public static TreeNode insertIntoAVL(final TreeNode node, final int key, final int count[], final int index) {
	    if (node == null) {
	        return new TreeNode(key);
	    }

	    if (node.key > key) {
	        node.left = insertIntoAVL(node.left, key, count, index);
	    } else {
	        node.right = insertIntoAVL(node.right, key, count, index);

	        // update smaller elements count
	        count[index] = count[index] + size(node.left) + 1;
	    }

	    // update the size and height
	    node.height = max(height(node.left), height(node.right)) + 1;
	    node.size = size(node.left) + size(node.right) + 1;

	    // balance the tree
	    final int balance = height(node.left) - height(node.right);
	    // left-left
	    if (balance > 1 && node.key > key) {
	        return rotateRight(node);
	    }
	    // right-right
	    if (balance < -1 && node.key > key) {
	        return rotateLeft(node);
	    }
	    // left-right
	    if (balance > 1 && node.key < key) {
	        node.left = rotateLeft(node.left);
	        return rotateRight(node);
	    }
	    // right-left
	    if (balance > 1 && node.key < key) {
	        node.right = rotateRight(node.right);
	        return rotateLeft(node);
	    }

	    return node;
	}

	public static int[] countSmallerOnRight(final int[] in) {
	    final int[] smaller = new int[in.length];

	    TreeNode root = null;
	    for (int i = in.length - 1; i >= 0; i--) {
	        root = insertIntoAVL(root, in[i], smaller, i);
	    }

	    return smaller;
	}
	
	public static void rotateRight(int[] A, int i, int j){
		int temp = A[j];
		System.arraycopy(A, i, A, i+1, j-i);
		A[i] = temp;
	}
	
	public static void mergeInPlace(int[] A, int i, int j){
		while(i < j && i < A.length && j < A.length){
			if(A[i] > A[j]){
				rotateRight(A, i, j);
				i++;
				j++;
			}
			else{
				i++;
			}
		}
	}
	
	public static void mergeSortInPlace(int[] A, int i, int j){
		if(i >= j){
			return;
		}
		int k  = (i+j)/2;
		
		mergeSortInPlace(A, i, k);
		mergeSortInPlace(A, k+1, j);
		mergeInPlace(A, i, k+1);
	}
	
	
	public static void merge(int[] A, int i, int j, int k){
		int[] B = new int[A.length];
		System.arraycopy(A, 0, B, 0, A.length);
		
		for(int r = 0; r <= k && i<A.length && j<A.length; r++){
			if(B[i] > B[j]){
				A[r] = B[j++];
			}
			else{
				A[r] = B[i++];
			}
		}
		System.out.println("");
	}
	
	public static void mergeSort(int[] A, int i, int j){
		if(i >= j){
			return;
		}
		int k  = (i+j)/2;
		
		mergeSort(A, i, k);
		mergeSort(A, k+1, j);
		merge(A, i, k+1, A.length-1);
	}
	
	private int minLenSumPathBST(final TreeNode root, final int sum, final int len) {
	    if (root == null) {
	        return Integer.MAX_VALUE;
	    }
	
	    // find the remaining sum as we are including current node in the current path
	    final int remainingSum = sum - root.key;
	    // If remaining sum is zero and it is A leaf node then we found A complete path from root to A leaf.
	    if (remainingSum == 0 && root.left == null && root.right == null) {
	        return len + 1;
	    }
	    // If remaining sum is less than current node value then we search remaining in the left subtree.
	    else if (remainingSum <= root.key) {
	    	 int l = minLenSumPathBST(root.left, remainingSum, len + 1);
	         // if search in left subtree fails to find such path only then we search in the right subtree
	         if (l == Integer.MAX_VALUE) {
	             l = minLenSumPathBST(root.right, remainingSum, len + 1);
	         }
	
	         return l;
	        
	    }
	    // If remaining sum is greater than current node value then we search remaining in the right subtree.
	    else {
	    	 int l = minLenSumPathBST(root.right, remainingSum, len + 1);
	         // if search in right subtree fails to find such path only then we search in the left subtree
	         if (l == Integer.MAX_VALUE) {
	             l = minLenSumPathBST(root.left, remainingSum, len + 1);
	         }
	
	         return l;
	    }
	}
	
	
	public String[] topKWords(final String stream, final int k) {
	    final class WordFreq implements Comparable<WordFreq> {
	        String word;
	        int freq;
	
	        public WordFreq(final String w, final int c) {
	            word = w;
	            freq = c;
	        }
	
	        @Override
	        public int compareTo(final WordFreq other) {
	            return Integer.compare(this.freq, other.freq);
	        }
	    }
	    final Map<String, Integer> frequencyMap = new HashMap<String, Integer>();
	    final PriorityQueue<WordFreq> topKHeap = new PriorityQueue<WordFreq>(k);
	
	    final String[] words = stream.toLowerCase().trim().split(" ");
	    for (final String word : words) {
	        int freq = 1;
	        if (frequencyMap.containsKey(word)) {
	            freq = frequencyMap.get(word) + 1;
	        }
	
	        // update the frequency map
	        frequencyMap.put(word, freq);
	    }
	
	    // Build the topK heap
	    for (final java.util.Map.Entry<String, Integer> entry : frequencyMap.entrySet()) {
	        if (topKHeap.size() < k) {
	            topKHeap.add(new WordFreq(entry.getKey(), entry.getValue()));
	        } else if (entry.getValue() > topKHeap.peek().freq) {
	            topKHeap.remove();
	            topKHeap.add(new WordFreq(entry.getKey(), entry.getValue()));
	        }
	    }
	
	    // extract the top K
	    final String[] topK = new String[k];
	    int i = 0;
	    while (topKHeap.size() > 0) {
	        topK[i++] = topKHeap.remove().word;
	    }
	
	    return topK;
	}
	
	public static void swap(final int[] a, final int i, final int j) {
	    if (i == j || i < 0 || j < 0 || i > a.length - 1 || j > a.length - 1) {
	        return;
	    }
	    a[i] ^= a[j];
	    a[j] ^= a[i];
	    a[i] ^= a[j];
	}
	
	public static int[] nextEven(final int[] digits) {
	    int y = digits.length - 1;
	    boolean evenFound = digits[y] % 2 == 0;
	    // find longest increasing subarray from right to left
	    for (int i = digits.length - 2; i >= 0; i--) {
	        if (digits[i] >= digits[i + 1]) {
	            evenFound |= digits[i] % 2 == 0;
	            y = i;
	        } else {
	            break;
	        }
	    }
	
	    int maxEven = -1;
	    // if y doesn’mergedRank contain an even then extend y to left until an even found
	    while (!evenFound && y - 1 >= 0 && digits[y - 1] % 2 != 0) {
	        y--;
	    }
	
	    // input is already the largest permutation
	    if (y <= 0) {
	        return digits[digits.length - 1] % 2 == 0 ? digits : null;
	    }
	
	    //try to extend Y such that y contains an even after swapping X[A] with the Y[rank]
	    while(y -1 >= 0){
		    // now X = digits[0..y-1], and Y = digits[y..digits.length-1]
		    // A is the rightmost element of x, i.e. A = y-1;
		    // find rank = min of y greater than A
		    final int a = y - 1;
		    int b = -1;
		    for (int i = y; i < digits.length; i++) {
		        b = digits[i] > digits[a] && (b == -1 || (digits[i] < digits[b])) ? i : b;
		    }
	
		    // input is already the largest permutation
		    if (b == -1) {
		        return digits[digits.length - 1] % 2 == 0 ? digits : null;
		    }
		    // swap A and rank
		    swap(digits, a, b);
	
		    // update max even in y
		    for (int i = y; i < digits.length; i++) {
		        maxEven = digits[i] % 2 == 0 && (maxEven == -1 || (maxEven != -1 && digits[i] > digits[maxEven])) ? i
		                : maxEven;
		    }
	
		    // input is already the largest permutation or need to extend y
		    if (maxEven == -1) {
		    	y--;
		    }
		    else{
		    	break;
		    }
	    }
		    
	    if (maxEven == -1) {
	        return digits[digits.length - 1] % 2 == 0 ? digits : null;
	    }
	
	    // swap max even with rightmost position
	    swap(digits, maxEven, digits.length - 1);
	    // sort y leaving rightmost position unchanged
	    Arrays.sort(digits, y, digits.length - 1);
	
	    return digits;
	}
	
	public static boolean isCrossed(double[]  s){
		//base case 
		if(s.length < 4){
			return false;
		}
		if(s[0] >= s[2] && s[3] >= s[1]){
			return true;
		}
		
		//test if the moves are on outward increasing spiral
		int i = 3;
		while(i < s.length){
			if(s[i] > s[i-2] && s[i-1] > s[i-3])
				i++;
			else break;
		}
		
		//if we visited all the moves then there is no intersection
		if(i == s.length){
			return false;
		}
		
		//otherwise moves are on A decreasing inward spiral starting from i
		//we first need check if the two spirals are crossing each other which can only possible
		//when edge i+1 crosses edge (i-4)  or edge i+1 crosses edge i-2 (if exists)
		
		if(i < s.length && i > 3 && s[i + 1] >= (s[i - 1] - s[i - 3])){
			if (s[i] >= (s[i - 2] - s[i - 4]) || s[i + 1] >= s[i - 1])
	    		return true;
		}
		
		//if two spiral didn'mergedRank intersect then check for decreasing s
		while(i+3 < s.length){
			if(s[i] > s[i+2] && s[i+1] > s[i+3]){
				i++;
			}
			else break;
		}
		
		//if we visited all the moves then there is no intersection
		if(i+3 == s.length){
			return false;
		}
		
		return false;
	}
	
	// insert current element to the left or right heap and get median so far
	public static double getMedian(final int current, final double med, final PriorityQueue<Integer> left, final PriorityQueue<Integer> right) {
		final int balance = left.size() - right.size();
	    double median = med;
	
	    // both heaps are of equal size.
	    if (balance == 0) {
	        // need to insert in left
	        if (current < median) {
	            left.offer(current);
	            median = left.peek();
	        }
	        // need to insert in right
	        else {
	            right.offer(current);
	            median = right.peek();
	        }
	    }
	    // left heap is larger
	    else if (balance > 0) {
	        // need to insert in left
	        if (current < median) {
	            right.offer(left.poll());
	            left.offer(current);
	        }
	        // need to insert in right
	        else {
	            right.offer(current);
	        }
	
	        median = (left.peek() + right.peek()) / 2.0;
	    }
	    // right heap is larger
	    else if (balance < 0) {
	        // need to insert in left
	        if (current < median) {
	            left.offer(current);
	        }
	        // need to insert in right
	        else {
	            left.offer(right.poll());
	            right.offer(current);
	        }
	
	        median = (left.peek() + right.peek()) / 2.0;
	    }
	
	    return median;
	}
	
	public static double getStreamMedian(final int[] stream) {
	    double median = 0;
	    final PriorityQueue<Integer> left = new PriorityQueue<Integer>(16, Collections.reverseOrder());
	    final PriorityQueue<Integer> right = new PriorityQueue<Integer>(16);
	
	    for (int i = 0; i < stream.length; i++) {
	        median = getMedian(stream[i], median, left, right);
	    }
	    return median;
	}
	
	public static boolean isSubSetSum(final int[] set, final int sum) {
	    final int m = set.length;
	    final boolean[][] ssTable = new boolean[sum + 1][m + 1];

	    // base cases: if m == 0 then no solution for any sum
	    for (int i = 0; i <= sum; i++) {
	        ssTable[i][0] = false;
	    }
	    // base case: if sum = 0 then there is only one solution for any input set: just take none of each of the items.
	    for (int j = 0; j <= m; j++) {
	        ssTable[0][j] = true;
	    }

	    for (int i = 1; i <= sum; i++) {
	        for (int j = 1; j <= m; j++) {
	            // solutions excluding last element i.e. set[j-1]
	            final boolean s1 = ssTable[i][j - 1];
	            // solutions including last element i.e. set[j-1]
	            final boolean s2 = (i - set[j - 1]) >= 0 ? ssTable[i - set[j - 1]][j - 1] : false;

	            ssTable[i][j] = s1 || s2;
	        }
	    }

	    return ssTable[sum][m];
	}
	
	//overall O(n^2) time and O(n) space solution using A greedy approach
	public static ArrayList<Integer>[] findEqualPartitionMinSumDif(int A[]){
		//first sort the array - O(nlgn)
		Arrays.sort(A);
		ArrayList<Integer> partition1 = new ArrayList<Integer>();
		ArrayList<Integer> partition2 = new ArrayList<Integer>();
		
		//create index table to manage largest unused and smallest unused items
		//O(n) space and O(nlgn) time to build and query the set
		TreeSet<Integer> unused = new TreeSet<>();
		for(int i = 0; i<A.length; i++){
			unused.add(i);
		}
		
		int i = 0;
		int j = A.length-1;
		int part1Sum = 0;
		int part2Sum = 0;
		int diffSum = 0;
		
		//O(n^2) processing time
		while(unused.size() > 0){
			i = unused.first();
			j = unused.last();
			diffSum = part1Sum-part2Sum;
			
			//in case of size of the array is not multiple of 4 then we need to process last 3(or 2 or 1)
			//element to assign partition. This is special case handling
			if(unused.size() < 4){
				switch(unused.size()){
					case 1: 
						//put the 1 remaining item into smaller partition
						if(diffSum > 0){
							partition2.add(A[i]);
							part2Sum += A[i];
						}
						else{
							partition1.add(A[i]);
							part1Sum += A[i];
						}
					break;
					case 2:
						//among the remaining 2 put the max in smaller and min in larger bucket
						int max = Math.max(A[i], A[j]);
						int min = Math.min(A[i], A[j]);
						if(diffSum > 0){
							partition2.add(max);
							partition1.add(min);
							part2Sum += max;
							part1Sum += min;
						}
						else{
							partition1.add(max);
							partition2.add(min);
							part1Sum += max;
							part2Sum += min;
						}
					break;
					case 3:
						//among the remaining 3 put the two having total value greater then the third one into smaller partition
						//and the 3rd one to larger bucket 
						unused.remove(i);
						unused.remove(j);
						int middle = unused.first();
						
						if(diffSum > 0){
							if(A[i]+A[middle] > A[j]){
								partition2.add(A[i]);
								partition2.add(A[middle]);
								partition1.add(A[j]);
								part2Sum += A[i]+A[middle];
								part1Sum += A[j];
							}
							else{
								partition2.add(A[j]);
								partition1.add(A[i]);
								partition1.add(A[middle]);
								part1Sum += A[i]+A[middle];
								part2Sum += A[j];
							}
						}
						else{
							if(A[i]+A[middle] > A[j]){
								partition1.add(A[i]);
								partition1.add(A[middle]);
								partition2.add(A[j]);
								part1Sum += A[i]+A[middle];
								part2Sum += A[j];
							}
							else{
								partition1.add(A[j]);
								partition2.add(A[i]);
								partition2.add(A[middle]);
								part2Sum += A[i]+A[middle];
								part1Sum += A[j];
							}
						}
					break;
					default:
				}
				
				diffSum = part1Sum-part2Sum;
				break;
			}
			
			//first take the largest and the smallest element to create A pair to be inserted into A partition
			//we do this for having A balanced distribute of the numbers in the partitions
			//add pair (i, j) to the smaller partition 
			int pairSum = A[i]+A[j];
			int partition = diffSum > 0 ? 2 : 1;
			if(partition == 1){
				partition1.add(A[i]);
				partition1.add(A[j]);
				part1Sum += pairSum;
			}
			else{
				partition2.add(A[i]);
				partition2.add(A[j]);
				part2Sum += pairSum;
			}
			
			//update diff
			diffSum = part1Sum-part2Sum;
			//we have used pair (i, j)
			unused.remove(i);
			unused.remove(j);
			//move j to next big element to the left
			j = unused.last();
			//now find the buddy for j to be paired with such that sum of them is as close as to pairSum
			//so we will find such buddy A[k], i<=k<j such that value of ((A[j]+A[k])-pairSum) is minimized.
			int buddyIndex = unused.first();
			int minPairSumDiff = Integer.MAX_VALUE;
			for(int k = buddyIndex; k<j; k++){
				if(!unused.contains(k))
					continue;
				
				int compPairSum = A[j]+A[k];
				int pairSumDiff = Math.abs(pairSum-compPairSum);
				
				if(pairSumDiff < minPairSumDiff){
					minPairSumDiff = pairSumDiff;
					buddyIndex = k;
				}
			}
			
			//we now find buddy for j. So we add pair (j,buddyIndex) to the other partition
			if(j != buddyIndex){
				pairSum = A[j]+A[buddyIndex];
				if(partition == 2){
					partition1.add(A[j]);
					partition1.add(A[buddyIndex]);
					part1Sum += pairSum;
				}
				else{
					partition2.add(A[j]);
					partition2.add(A[buddyIndex]);
					part2Sum += pairSum;
				}
				
				//we have used pair (j, buddyIndex)
				unused.remove(j);
				unused.remove(buddyIndex);
			}
		}
		
		//if diffsum is greater than zero then we can further try to optimize by swapping 
		//A larger elements in large partition with an small element in smaller partition
		//O(n^2) operation with O(n) space
		if(diffSum != 0){
			Collections.sort(partition1);
			Collections.sort(partition2);
			
			diffSum = part1Sum-part2Sum;
			ArrayList<Integer> largerPartition = (diffSum > 0) ? partition1 : partition2;
			ArrayList<Integer> smallerPartition = (diffSum > 0) ? partition2 : partition1;
			
			int prevDiff = Math.abs(diffSum);
			int largePartitonSwapCandidate = -1;
			int smallPartitonSwapCandidate = -1;
			//find one of the largest element from large partition and smallest from the smaller partition to swap 
			//such that it overall sum difference in the partitions are minimized
			for(i = 0; i < smallerPartition.size(); i++){
				for(j = largerPartition.size()-1; j>=0; j--){
					int largerVal = largerPartition.get(j);
					int smallerVal = smallerPartition.get(i);
					
					//no point of swapping larger value from smaller partition
					if(largerVal <= smallerVal){
						continue;
					}
	
					//new difference if we had swapped these elements
					int diff = Math.abs(prevDiff - 2*Math.abs(largerVal - smallerVal));
					if(diff == 0){
						largerPartition.set(j, smallerVal);
						smallerPartition.set(i, largerVal);
						return new ArrayList[]{largerPartition, smallerPartition};
					}
					//find the pair to swap that minimizes the sum diff
					else if (diff < prevDiff){
						prevDiff = diff;
						largePartitonSwapCandidate = j;
						smallPartitonSwapCandidate = i;
					}
				}
			}
			
			//if we indeed found one such A pair then swap it. 
			if(largePartitonSwapCandidate >=0 && smallPartitonSwapCandidate >=0){
				int largerVal = largerPartition.get(largePartitonSwapCandidate);
				int smallerVal = smallerPartition.get(smallPartitonSwapCandidate);
				largerPartition.set(largePartitonSwapCandidate, smallerVal);
				smallerPartition.set(smallPartitonSwapCandidate, largerVal);
				return new ArrayList[]{largerPartition, smallerPartition};
			}
		}
		
		return new ArrayList[]{partition1, partition2};
	}
	
	public static int largestPlusInMatrix(int M[][]){
		int n = M.length;
		int m = M[0].length;
		int left[][] = new int[n+2][m+2];
		int right[][] = new int[n+2][m+2];
		int top[][] = new int[n+2][m+2];
		int bottom[][] = new int[n+2][m+2];
		
		//topdown
		for(int i = 1; i<= n; i++){
			for(int j = 1; j <= m; j++){
				left[i][j] = (M[i-1][j-1] == 0) ? 0 : left[i][j-1]+1;
				top[i][j] = (M[i-1][j-1] == 0) ? 0 : top[i-1][j]+1;
			}
		}
		
		//bottomup
		for(int i = n; i >= 1; i--){
			for(int j = m; j >= 1; j--){
				right[i][j] = (M[i-1][j-1] == 0) ? 0 : right[i][j+1]+1;
				bottom[i][j] = (M[i-1][j-1] == 0) ? 0 : bottom[i+1][j]+1;
			}
		}
		
		int minPlus[][] = new int[n][m];
		int maxPlusLen = -1;
		int maxPlusRow = -1;
		int maxPlusCol = -1;
		for(int i = 0; i< n; i++){
			for(int j = 0; j < m; j++){
				minPlus[i][j] = Math.min(Math.min(left[i+1][j+1], right[i+1][j+1]), Math.min(top[i+1][j+1], bottom[i+1][j+1]));
				
				if(minPlus[i][j] > maxPlusLen){
					maxPlusLen = minPlus[i][j];
					maxPlusRow = i;
					maxPlusCol = j;
				}
			}
		}
		
		System.out.println("[row,col]=["+maxPlusRow+","+maxPlusCol+"]");
		return (maxPlusLen-1)*4+1;
	}
	
	static class SerializableTree{
		public String value;
		public ArrayList<SerializableTree> children = new ArrayList<test.SerializableTree>();
		int childCount;
		
		public SerializableTree(){
		}
		
		public SerializableTree(String val){
			value = val;
		}
		
		public void addChild(SerializableTree child){
			children.add(child);
		}
		
		public SerializableTree(String val, SerializableTree[] childs){
			value = val;
			
			for(int i = 0; i<childs.length; i++){
				children.add(childs[i]);
			}
		}
		
		public static String serialize(SerializableTree root){
			StringBuilder serialized = new StringBuilder();
			Queue<SerializableTree> queue = new LinkedList<SerializableTree>();
			queue.offer(root);
			
			while(!queue.isEmpty()){
				SerializableTree node = queue.poll();
				int childrenCount = node.children.size();
				
				serialized.append(node.value);
				serialized.append(",");
				serialized.append(childrenCount);
				serialized.append("#");
				
				for(int i = 0; i<childrenCount; i++){
					SerializableTree child = node.children.get(i);
					queue.offer(child);
				}
			}
			
			return serialized.toString();
		}
		
		public static SerializableTree deserialize(String serialized){
			
			Queue<SerializableTree> queue = new LinkedList<SerializableTree>();
			String[] bfsNodes = serialized.split("#");
			String rootSer[] = bfsNodes[0].trim().split(",");
			
			SerializableTree root = new SerializableTree();
			root.value = rootSer[0].trim();
			root.childCount = Integer.parseInt(rootSer[1].trim());
			queue.offer(root);
			
			int serIndex = 1;
			while(!queue.isEmpty()){
				SerializableTree node = queue.poll();
				
				for(int i = 0; i< node.childCount; i++){
					String childSer[] = bfsNodes[serIndex+i].trim().split(",");
					
					SerializableTree child = new SerializableTree();
					child.value = childSer[0].trim();
					child.childCount = Integer.parseInt(childSer[1].trim());
					
					node.addChild(child);
					queue.offer(child);
				}
				serIndex += node.childCount;
			}
			
			return root;
		}
	}
	
	private static void kPermutation(String pref, String str, int k){
		if(k == 0){
			System.out.print(pref+", ");
		}
		else{
			for(int i = 0; i<str.length(); i++){
				kPermutation(pref+str.charAt(i), str.substring(0, i)+str.substring(i+1), k-1);
			}
		}
	}
	
	private static void PermutationWithRepts(String pref, String str, int k){
		if(k == 0){
			System.out.print(pref+", ");
		}
		else{
			for(int i = 0; i<str.length(); i++){
				PermutationWithRepts(pref+str.charAt(i), str, k-1);
			}
		}
	}
	
	public static void permutation(String str, int k, boolean repetitionAllowed){
		if(k > 0 && k <= str.length()){
			if(!repetitionAllowed)
				kPermutation("", str, k);
			else
				PermutationWithRepts("", str, k);
			System.out.print("\n");
		}
	}
	
	private static void uniquePermutation(String pref, String str, int k, Set<String> visited){
		if(!visited.contains(pref)){
			if(k == 0){
				System.out.print(pref+", ");
			}
			else{
				for(int i = 0; i<str.length(); i++){
					uniquePermutation(pref+str.charAt(i), str.substring(0, i)+str.substring(i+1), k-1, visited);
					visited.add(pref+str.charAt(i));
				}
			}
		}
	}
	
	public static void uniquePermutation(String str, int k){
		if(k > 0 && k <= str.length()){
			Set<String> visited = new HashSet<String>();
			uniquePermutation("", str, k, visited);
			System.out.print("\n");
		}
	}
	
	private static void kCombination(String pref, String str, int k){
		if(k == 0){
			System.out.print(pref+", ");
		}
		else{
			for(int i = 0; i<str.length(); i++){
				kCombination(pref+str.charAt(i), str.substring(i+1), k-1);
			}
		}
	}
	
	public static void combination(String str, int k){
		if(k > 0 && k <= str.length()){
			kCombination("", str, k);
			System.out.print("\n");
		}
	}
	
	private static void allCombination(String pref, String str){
		if(!pref.isEmpty())
			System.out.print(pref+", ");
		for(int i = 0; i<str.length(); i++){
			allCombination(pref+str.charAt(i), str.substring(i+1));
		}
	}
	
	public static void allCombination(String str){
		allCombination("", str);
		System.out.print("\n");
	}

	public static TreeNode[] randomKSampleTreeNode(TreeNode root, int k){
		TreeNode[] reservoir = new TreeNode[k];
		Queue<TreeNode> queue = new LinkedList<TreeNode>();
		queue.offer(root);
		int index = 0;
		
		//copy first k elements into reservoir 
		while(!queue.isEmpty() && index < k){
			TreeNode node = queue.poll();
			reservoir[index++] = node;
			if(node.left != null){
				queue.offer(node.left);
			}
			if(node.right != null){
				queue.offer(node.right);
			}
		}
		
		//for index k+1 to the last node of the tree select random index from (0 to index) 
		//if random index is less than k than replace reservoir node at this index by 
		//current node
		while(!queue.isEmpty()){
			TreeNode node = queue.poll();
			int j = (int) Math.floor(Math.random()*(index+1));
			index++;
			
			if(j < k){
				reservoir[j] = node;
			}
		
			if(node.left != null){
				queue.offer(node.left);
			}
			if(node.right != null){
				queue.offer(node.right);
			}
		}
		
		return reservoir;
	}
	
	public static double findMedianSortedArrays1(int A[], int B[]) {
		int m = A.length;
		int n = B.length;
	 
		if ((m + n) % 2 != 0) // odd
			return (double) findKth(A,0, m - 1, B, 0, n - 1, (m + n) / 2);
		else { // even
			return (findKth(A, 0, m - 1, B, 0, n - 1, (m + n) / 2) 
				+ findKth(A, 0, m - 1, B, 0, n - 1, (m + n) / 2 - 1)) * 0.5;
		}
	}
	
	public static int findKth(int A[], int p1, int r1, int B[], int p2, int r2, int k) {
		int n1 = r1-p1+1;
		int n2 = r2-p2+1;
		
		//base cases
		if(n1 == 0){
			return B[p2+k];
		}
		if(n2 == 0){
			return A[p1+k];
		}
		//
		if(k == 0){
			return Math.min(A[p1], B[p2]);
		}
		
		//select two index i,j from A and B respectively such that If A[i] is between B[j] and B[j-1] 
		//Then A[i] would be the i+j+1 smallest element because.
		//Therefore, if we choose i and j such that i+j = k-1, we are able to find the k-th smallest element.
		int i = (n1*k)/(n1+n2);//let's try tp chose a middle element close to kth element in A 
		int j = k-1 -i;
		
		//add the offset
		int mid1 = Math.min(p1+i, r1);
		int mid2 = Math.min(p2+j, r2);
		
		//mid1 is greater than mid2. So, median is either in A[p1...mid1] or in B[mid2+1...r2].
		//we have already see B[p2..mid2] elements smaller than kth smallest
		if(A[mid1] > B[mid2]){
			k = k - (mid2-p2+1);
			r1 = mid1;
			p2 = mid2+1;
		}
		//mid2 is greater than or equal mid1. So, median is either in A[mid1+1...r1] or in B[p2...mid2].
		//we have already see A[p1..mid1] elements smaller than kth smallest
		else{
			k = k - (mid1-p1+1);
			p1 = mid1+1;
			r2 = mid2;
		}
		
		return findKth(A, p1, r1, B, p2, r2, k);
	}
	 
//	public static int findKth(int A[], int B[], int k, 
//		int aStart, int aEnd, int bStart, int bEnd) {
//	 
//		int aLen = aEnd - aStart + 1;
//		int bLen = bEnd - bStart + 1;
//	 
//		// Handle special cases
//		if (aLen == 0)
//			return B[bStart + k];
//		if (bLen == 0)
//			return A[aStart + k];
//		if (k == 0)
//			return A[aStart] < B[bStart] ? A[aStart] : B[bStart];
//	 
//		int aMid = aLen * k / (aLen + bLen); // a's middle count
//		int bMid = k - aMid - 1; // b's middle count
//	 
//		// make aMid and bMid to be array index
//		aMid = aMid + aStart;
//		bMid = bMid + bStart;
//	 
//		if (A[aMid] > B[bMid]) {
//			k = k - (bMid - bStart + 1);
//			aEnd = aMid;
//			bStart = bMid + 1;
//		} else {
//			k = k - (aMid - aStart + 1);
//			bEnd = bMid;
//			aStart = aMid + 1;
//		}
//	 
//		return findKth(A, B, k, aStart, aEnd, bStart, bEnd);
//	}
	
    public static double median(int[] A, int[] B){
    	int n = A.length;
    	int m = B.length;
    	
    	if((n+m)%2 == 0){
    		double mid1 = kthSmallestElement(A, 0, A.length-1, B, 0, B.length-1, n/2-1);
    		double mid2 = kthSmallestElement(A, 0, A.length-1, B, 0, B.length-1, n/2);
    		return (mid1+mid2)/2;
    	}
    	else{
    		return kthSmallestElement(A, 0, A.length-1, B, 0, B.length-1, n/2);
    	}
    }
	
	public static int kthSmallestElement(int[] A, int p1, int r1, int[] B, int p2, int r2, int k){
		
		//base cases
		//if A is exhausted so return B's kth smallest
		if(p1 > r1){
			return B[p2+k];
		}
		//or if B is exhausted so return A's kth smallest
		if(p2 > r2){
			return A[p1+k];
		}
		
		//middle points
		int q1 = (p1+r1)/2;
		int q2 = (p2+r2)/2;
		//left partition sizes
		int m1 = q1-p1+1;
		int m2 = q2-p2+1;
		
		//combination left partition doesn'mergedRank include kth smallest
		if(m1+m2 < k){
			//left partition of B is smaller than kth smallest, so discard it.
			//we are discarding  m2 smaller elements, so search for (k-m2)th smallest 
			if(A[m1] > B[m2]){
				return kthSmallestElement(A, p1, r1, B, q2+1, r2, k-m2);
			}
			//left partition of A is smaller than the kth smallest, so discard it
			//we are discarding  m1 smaller elements, so search for (k-m1)th smallest 
			else{
				return kthSmallestElement(A, q1+1, r1, B, p2, r2, k-m1);
			}
		}
		else{
			//right partition of A is larger than kth smallest, so discard it.
			if(A[m1] > B[m2]){
				return kthSmallestElement(A, p1, q1-1, B, q2+1, r2, k);
			}
			//right partition of B is larger than the kth smallest, so discard it
			else{
				return kthSmallestElement(A, q1, r1, B, p2, q2-1, k);
			}
		}
	}
    
//    public static int kthSmallestElement(int[] A, int p1, int r1, int[] B, int p2, int r2, int k){
//    	int n1 = r1-p1+1;
//    	int n2 = r2-p2+1;
//    	
//    	//invariant 1 : n1 < n2
//    	if(n1 > n2){
//    		kthSmallestElement(B, p2, r2, A, p1, r1, k);
//    	}
//    	//base cases
//    	if(n1 == 0 && n2 > 0){
//    		return B[k-1];
//    	}
//    	if(k == 1){
//    		return Math.min(A[0], B[0]);
//    	}
//    	
//    	//divide and conquer
//    	int i = Math.min(n1, k/2);
//    	int j = Math.min(n2, k/2);
//    	
//    	//kth smallest is in left part of k/2th index in A and right part of k/2th index in B 
//    	//Also, j elements in B's left part are definitely smaller than kth smallest
//    	//so, we already see j smaller elements and we need to find (k-j)th element 
//    	if(A[i-1] > B[j-1]){
//    		return kthSmallestElement(A, p1, p1+i-1, B, p2+j-1, r2, k-j);
//    	}
//    	else{
//    		return kthSmallestElement(A, p1+i-1, p2, B, p2, p2+j-1, k-i);
//    	}
//    }
	
	public static int median(int[] a, int p, int r){
		int n = r-p+1;
		if(n%2 == 0){
			return (a[p+n/2]+a[p+n/2-1])/2;
		}
		else{
			return a[p+n/2];
		}
	}
	
	public static int median(int a1[], int p1, int r1, int a2[], int p2, int r2){
		int n = r1-p1+1;
		
		//base cases
		if(n<=0){
			return Integer.MIN_VALUE;
		}
		else if(n == 1){
			return (a1[0]+a2[0])/2;
		}
		else if(n == 2){
			return (Math.max(a1[0], a2[0]) + Math.min(a1[1], a2[1]))/2;
		}
		
		int m1 = median(a1, p1, r1);
		int m2 = median(a2, p2, r2);
		
		if(m1 == m2){
			return m1;
		}
		
		//median located in a1[m1..r1] and a2[p2...m2]
		if(m1 < m2){
			if(n%2 == 0){
				return median(a1, p1+n/2-1, r1, a2, p2, r2-n/2+1);
			}
			else{
				return median(a1, p1+n/2, r1, a2, p2, r2-n/2);
			}
		}
		//median located in a1[p1..m1] and a2[m2..r2]
		else{
			if(n%2 == 0){
				return median(a1, p1, r2-n/2+1, a2, p2+n/2-1, r2);
			}
			else{
				return median(a1, p1, r2-n/2, a2, p2+n/2, r2);
			}
		}
	}
	
	public static int median(int[][] A){
		int n = A.length;
		int m = A[0].length;
		
		if((n*m)%2 == 0){
			int mid1 = kthSmallestElement(A, n/2-1);
			int mid2 = kthSmallestElement(A, n/2+1);
			return (mid1+mid2)/2;
		}
		else{
			return kthSmallestElement(A, n/2);
		}
	}
	
	public static int kthSmallestElement(int[][] A, int k){
		int n = A.length;
		int m = A[0].length;
		MatrixElement kthSmallest = null;
		
		PriorityQueue<MatrixElement> minHeap = new PriorityQueue<MatrixElement>();
		
		//add column 0 into meanHeap - O(nlgn)
		for(int i = 0; i<n; i++){
			minHeap.offer(new MatrixElement(A[i][0], i, 0));
		}
		
		//extract min from minheap and insert next element from the same row of the extracted min
		int count = 0;
		while(!minHeap.isEmpty() && count < k){
			kthSmallest = minHeap.poll();
			count++;
			//
			if(kthSmallest.col+1 < m){
				minHeap.offer(new MatrixElement(A[kthSmallest.row][kthSmallest.col+1], kthSmallest.row, kthSmallest.col+1));
			}
		}
		
		return kthSmallest.val;
	}
	
	public static class MatrixElement implements Comparable<MatrixElement>{
		public int val;
		public int row;
		public int col;
		
		public MatrixElement(int val, int row, int col){
			this.val = val;
			this.row = row;
			this.col = col;
		}
		@Override
		public int compareTo(MatrixElement o) {
			return Integer.compare(this.val, o.val);
		}
	}
	
	public static int findCLosestBST(TreeNode node, int key, int minDiff, int bestResult){
		int diff = Math.abs(node.key-key);
		
		if(diff < minDiff){
			minDiff = diff;
			bestResult = node.key;
		}
		
		if(minDiff == 0){
			return bestResult;
		}
		
		if(key < node.key && node.left != null){
			return findCLosestBST(node.left, key, minDiff, bestResult);
		}
		else if(key > node.key && node.right != null){
			return findCLosestBST(node.right, key, minDiff, bestResult);
		}
		else{
			return bestResult;
		}
	}
	
	public static void mergeToCountSmallerOnRight(int A[], int rank[], int p, int q, int r, int count[]){
		int n = r-p+1;
		int i = p;
		int j = q+1;
		int mid = q;
		int k=0;
		int mergedRank[] = new int[n];
		int smallerCount = 0;
		while(i <= mid && j <= r){
			//satisfies i<j, A[i]<A[j] -- so count smaller on right
			if(A[rank[i]] < A[rank[j]]){
				count[rank[i]] += smallerCount;
				mergedRank[k++] = rank[i++];
			}
			//i<j, A[i]>=A[j]
			else{
				smallerCount++;
				mergedRank[k++] = rank[j++];
			}
		}
		
		//copy leftovers from the two partitions into merge
		while(i <= mid){
			count[rank[i]] += r-q; 
			mergedRank[k++] = rank[i++];
		}
		while(j <= r){
			mergedRank[k++] = rank[j++];
		}
		
		//update rank
		System.arraycopy(mergedRank, 0, rank, p, n);
	}
	
	public static void countSmallerOnRightWithMerge(int A[], int rank[], int p, int r, int count[]){
		if(A.length == 1){
			return;
		}
		
		if(p < r){
			int q = (p+r)/2;
			//sort left side and count ic
			countSmallerOnRightWithMerge(A, rank, p, q, count);
			//sort right side and count ic
			countSmallerOnRightWithMerge(A, rank, q+1, r, count);
			//merge left and right and count cross ic
			mergeToCountSmallerOnRight(A, rank, p, q, r, count);
		}
	}
	
	public static int[] countSmallerOnRightWithMerge(int A[]){
		int n = A.length;
		int[] rank = new int[n];
		int count[] = new int[n];
		
		for(int i = 0; i < n; i++){
			rank[i] = i;
		}
		
		countSmallerOnRightWithMerge(A, rank, 0, n-1, count);
		
		return count;
	}
	
	//merge two sorted array A[0..q] and A[q+1..r] and return inversion count of each position 
	public static int mergeWithInvCount(int A[], int p, int q, int r){
		int crossInversionCount = 0;
		
		int n = r-p+1;
		int i = p;
		int j = q+1;
		int mid = q;
		int k=0;
		int merged[] = new int[n];
		while(i <= mid && j <= r){
			//satisfies i<j, A[i]<=A[j] -- so no inversion
			if(A[i] <= A[j]){
				merged[k++] = A[i++];
			}
			else{
				//i<j, A[i]>A[j] --- inversion count for A[j]
				crossInversionCount += (mid-i+1);
				merged[k++] = A[j++];
			}
		}
		
		//copy leftovers from the two partitions into merge
		while(i <= mid){
			merged[k++] = A[i++];
		}
		while(j <= r){
			merged[k++] = A[j++];
		}
		
		//update A
		System.arraycopy(merged, 0, A, p, n);
		
		return crossInversionCount;
	}
	
	public static int mergeSortWithInvCount(int A[], int p, int r){
		int inversionCount = 0;
		
		if(A.length == 1){
			return 0;
		}
		
		if(p < r){
			int q = (p+r)/2;
			//sort left side and count ic
			inversionCount = mergeSortWithInvCount(A, p, q);
			//sort right side and count ic
			inversionCount += mergeSortWithInvCount(A, q+1, r);
			
			//merge left and right and count cross ic
			inversionCount += mergeWithInvCount(A, p, q, r); 
		}
		
		return inversionCount;
	}
	
	public static class MaxSurpasser {
	    int[] A, rank, surp, mergedRank;
	
	    private MaxSurpasser(int[] a) {
	        this.A = a;
	        this.rank = new int[a.length];
	        this.surp = new int[a.length];
	        this.mergedRank = new int[a.length];
	        for (int i = 0; i < rank.length; i++){
	            rank[i] = i;
	        }
	    }
	
	    public static int find(int[] a) {
	        return new MaxSurpasser(a).sort();
	    }
	
	    private int sort() {
	        mergeSort(0, A.length - 1);
	        int max = 0;
	        System.out.print("bigger on rights count: ");
	        for (int i = 0; i < A.length; i++) {
	        	System.out.print(surp[i]+", ");
	            if (surp[i] > max) {
	                max = surp[i];
	            }
	        }
	        System.out.println();
	        return max;
	    }
	
	    private void mergeSort(int l, int r) {
	        if (l >= r) {
	            return;
	        }
	        //divide
	        int q = (l + r) / 2;
	        mergeSort(l, q);
	        mergeSort(q + 1, r);
	        //conquer
	        int i = l;
	        int j = q + 1; int acc = 0;
	        //accumulate through merge
	        for (int s = l; s <= r; s++) {
	        	if (j <= r && (i > q || A[rank[i]] < A[rank[j]])){
	                mergedRank[s] = rank[j];
	                acc++;
	                j++;
	            }
	        	else{
	        		mergedRank[s] = rank[i];
	                surp[rank[i]] += acc;
	                i++;
	        	}
	        }
	        for (int s = l; s <= r; s++) {
	            rank[s] = mergedRank[s];
	        }
	    }
	}
	
	//find the single number that duplicates one or more times in an array in O(1) space and O(n) time without modifying the array
	public static int findDuplicate(int[] nums) {
        //using Tortoise & Hair algorithm by Donald Knuth to find cycle in A sequence.
        //This algorithm also called Floyd's cycle detection algorithm
        int n = nums.length;
        int tortoise = n;
        int hair = n;
        
       do{
            tortoise = nums[tortoise-1];
            hair = nums[nums[hair-1]-1];
        } while(hair != tortoise);
        
        //find the starting point of the cycle and distance from the front, mu
        int mu = 0;
        tortoise = n;
        while(hair != tortoise){
            tortoise = nums[tortoise-1];
            hair = nums[hair-1];
            mu++;
        }
        
        //find the min length lambda of the cycle
        int lambda = 1;
        hair = nums[tortoise-1];
        
        while(hair != tortoise){
        	hair = nums[hair-1];
        	lambda++;
        }
        
        System.out.println("mu : "+mu+" lambda: "+lambda);
        
        return tortoise;
    }

	private static class Pair implements Comparable<Pair>{
		public int first;
		public int second;
		
		public Pair(int first, int second){
			this.first = first;
			this.second = second;
		}
	
		@Override
		public int compareTo(Pair o) {
			if(this.first == o.first)
				return Integer.compare(this.second, o.second);
			else
				return Integer.compare(this.first, o.first);
		}
	
		@Override
		public String toString() {
			return "[" + first + "," + second + "]";
		}
	}
	
	private static class Job implements Comparable<Job>{
		public int start;
		public int finish;
		public int weight;
		public int mode = 0;
		
		public Job(int start, int finish){
			this.start = start;
			this.finish = finish;
			this.weight = 1;
		}
		public Job(int start, int finish, int weight){
			this.start = start;
			this.finish = finish;
			this.weight = weight;
		}

		@Override
		public int compareTo(Job o) {
			if(mode == 1){
				return Integer.compare(this.finish, o.start);
			}
			else{
				return Integer.compare(this.finish, o.finish);
			}
		}

		@Override
		public String toString() {
			return "[" + start + "," + finish + "," + weight +"]";
		}
		
		public void print(){
			System.out.println(this.toString());
		}
	}
	
	//find exit path to the door
	public static void findExitWallsAndGates(int room[][]){
		Queue<Pair> queue = new LinkedList<Pair>();
		int n = room.length;
		int m = room[0].length;
		//down, right, up, left
		Pair[] dirs = {new Pair(1, 0), new Pair(0, 1), new Pair(-1, 0), new Pair(0, -1)};
		
		for(int i=0; i<room.length; i++){
			for(int j=0;j<room[0].length; j++){
				if(room[i][j] == 0){
					queue.offer(new Pair(i, j));
				}
			}
		}
		
		//BFS search
		while(!queue.isEmpty()){
			Pair pos = queue.poll();
			int r = pos.first;
			int c = pos.second;
			
			for(Pair dir : dirs){
				int i = r+dir.first;
				int j = c+dir.second;
				
				//prune the tree
				if(i < 0 || j < 0 || i>=n || j >= m || room[i][j] <= room[r][c]+1){
					continue;
				}
				
				room[i][j] = room[r][c]+1;
				queue.offer(new Pair(i, j));
			}
		}
	}

	//move zeros to left - in memory O(n)
    public void moveZeroes(int[] nums) {
        int left = -1, i = 0, n = nums.length;
        
        while(i < n){
            if(nums[i] != 0){
                if(left != -1){
                    swap(nums, i, left);
                    left++;
                }
            }
            else if(left == -1){
                    left = i;
            }
            
            i++;
        }
    }
    
    public List<String> addOperators2(String num, int target) {
        List<String> result = new ArrayList<String>();
        if (num.length() <= 1){
            return result;
        }
        int opnd1 = num.charAt(0) - '0';
        int opnd2 = num.charAt(1) - '0';
        String rem = (num.length() > 2) ? num.substring(2) : "";
        
        addOperators2(opnd1, opnd2, rem, target, "", result, -1, -1);  
        
        List<String> result2 = new ArrayList<String>();
        for(String res : result){
            StringBuffer sb = new StringBuffer();
            int i = 0;
            for(i = 0; i< num.length()-1; i++){
                sb.append(num.charAt(i));
                sb.append(res.charAt(i));
            }
            sb.append(num.charAt(i));
            
            result2.add(sb.toString());
        }
        return result2;
    }
    
    private void addOperators2(int opnd1, int opnd2, String remaining, int target, String path, List<String> assignment, int prevopnd1, int prevopnd2){
        char[] ops = {'+', '*', '-'};
        
        for(char op : ops){
            int newOprnd1 = -1;
            if(op == '+'){
                newOprnd1 = opnd1 + opnd2;
                path += "+";
            }
            else if(op == '*'){
                if(path.isEmpty() || path.endsWith("*")){
                    newOprnd1 = opnd1 * opnd2;
                }
                else if(path.endsWith("+")){
                    newOprnd1 = prevopnd1+(prevopnd2*opnd2);
                }
                else if(path.endsWith("-")){
                    newOprnd1 = prevopnd1-(prevopnd2*opnd2);
                }
                path += "*";
            }
            else if(op == '-'){
                newOprnd1 = opnd1 - opnd2;
                path += "-";
            }
            
            if(remaining.isEmpty()){
                if(newOprnd1 == target){
                    assignment.add(path);
                }
            }
            else{
                int newOprnd2 = remaining.charAt(0)-'0';
                String newRem = (remaining.length() > 1) ? remaining.substring(1) : "";
                addOperators2(newOprnd1, newOprnd2, newRem, target, path, assignment, opnd1, opnd2);
            }
            
            path = path.substring(0,path.length()-1);
        }
    }
    
    public List<String> addOperators(String num, int target) {
        List<String> result = new ArrayList<String>();
        if(num == null || num.length() == 0) return result;
         
        addOperator(num, 0, "", target, 0, 0, result);
        return result;
    }
    
    private void addOperator(String num, int curPos, String curPath, int target, long curEval, long prevOpnd, List<String> res){
        if(curPos == num.length()){
            if(curEval == target){
                res.add(curPath);
            }
            return;
        }
        
        for(int i = curPos; i<num.length(); i++){
            if(i != curPos && num.charAt(curPos) == '0'){
                break;
            }
            
            long curOprnd = Long.parseLong(num.substring(curPos, i+1));
            if(curPos == 0){
                addOperator(num, i+1, curPath+curOprnd, target, curOprnd, curOprnd, res);
            }
            else{
                addOperator(num, i+1, curPath+"+"+curOprnd, target, curEval+curOprnd, curOprnd, res);
                addOperator(num, i+1, curPath+"-"+curOprnd, target, curEval-curOprnd, -curOprnd, res);
                addOperator(num, i+1, curPath+"*"+curOprnd, target, curEval-prevOpnd+prevOpnd*curOprnd, prevOpnd*curOprnd, res);
            }
        }
    }
    
    /**
     * Definition for singly-linked list.
     **/
	  public static class ListNode {
	      int val;
	      ListNode next;
	      ListNode prev;
	      ListNode(int x) { val = x; }
	      
	      public void print(ListNode head){
	    	  while(head != null){
	    		  System.out.print(head.val+"->");
	    		  head = head.next;
	    	  }
	    	  System.out.println();
	      }
	      
	      public void printAsTree(ListNode head){
	    	  if(head == null){
	    		  return;
	    	  }
	    	  
	    	  printAsTree(head.prev);
	    	  System.out.print(head.val+", ");
	    	  printAsTree(head.next);
	      }
	  }
    
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode result = null;
        ListNode tail = null;
        int carry = 0;
        while(l1 != null || l2 != null){
            int a = (l1 != null) ? l1.val : 0;
            int b = (l2 != null) ? l2.val : 0;
            
            int sum = (a+b)+carry;
            carry = sum/10;
            ListNode node = new ListNode(sum%10);
            if(result == null){
                result = node;
                tail = node;
            }
            else{
                tail.next = node;
                tail=node;
            }
            
            l1 = l1 != null ? l1.next : l1;
            l2 = l2 != null ? l2.next : l2;
        }
        
        if(carry == 1){
            ListNode node = new ListNode(1);
            tail.next = node;
        }
        
        return result;
    }
    
    public static int lengthOfLongestNonrepeatedSubstring(String s) {
        int lastIndices[] = new int[256];
        for(int i = 0; i<256; i++){
            lastIndices[i] = -1;
        }
        
        int maxLen = 0;
        int curLen = 0;
        int start = 0;
        int bestStart = 0;
        for(int i = 0; i<s.length(); i++){
            char cur = s.charAt(i);
            if(lastIndices[cur]  < start){
                lastIndices[cur] = i;
                curLen++;
            }
            else{
                int lastIndex = lastIndices[cur];
                start = lastIndex+1;
                curLen = i-start+1;
                lastIndices[cur] = i;
            }
            
            if(curLen > maxLen){
                maxLen = curLen;
                bestStart = start;
            }
        }
        
        return maxLen;
    }
    
	//find number of sub arrays with A<=sum<=rank 
	public static int subArrayWithSumInRange(int[] A, int a, int b){
		int count = 0;
		TreeSet<Pair> treeSet = new TreeSet<test.Pair>();
		int cumsum = 0;
		
		for(int i = 0; i< A.length; i++){
			cumsum+=A[i];
			
			if(A[i] >= a && A[i] <= b){
				count++;
			}
			else if(cumsum >= a && cumsum <= b){
				count++;
			}
	
			NavigableSet<Pair> subSet = treeSet.subSet(new Pair(cumsum-b, i+1), true, new Pair(cumsum-a, i+1), false);
			if(!subSet.isEmpty()){
				count += Math.abs(subSet.first().second - subSet.last().second)+1;
			}
			treeSet.add(new Pair(cumsum, i));
		}
		
		return count;
	}
	
	//find number of sub arrays with A<=sum<=rank 
		public static int subArrayWithSumInRange1(int[] A, int a, int b){
			int count = 0;
			TreeMap<Pair, Integer> treeMap = new TreeMap<Pair, Integer>();
			int cumsum = 0;
			
			for(int i = 0; i< A.length; i++){
				cumsum+=A[i];
				
				if(A[i] >= a && A[i] <= b){
					count++;
				}
		
				NavigableMap<Pair, Integer> subMap = treeMap.subMap(new Pair(cumsum-b, -1), true, new Pair(cumsum-a, -1), true);
				if(!subMap.isEmpty()){
					count += Math.abs(subMap.firstEntry().getValue() - subMap.lastEntry().getValue())+1;
				}
				treeMap.put(new Pair(cumsum, i), i);
			}
			
			return count;
		}
	
	public static int findKthSmallest(int A[], int p1, int r1, int m, int B[], int p2, int r2, int n, int k) {
		  assert(m >= 0); assert(n >= 0); assert(k > 0); assert(k <= m+n);
		  
		  int i = p1+(int)((double)m / (m+n) * (k-1));
		  int j = p2+(k-1) - i;
		 
		  assert(i >= 0); assert(j >= 0); assert(i <= m); assert(j <= n);
		  // invariant: i + j = k-1
		  // Note: A[-1] = -INF and A[m] = +INF to maintain invariant
		  int Ai_1 = ((i <= 0) ? Integer.MIN_VALUE : A[i-1]);
		  int Bj_1 = ((j <= 0) ? Integer.MIN_VALUE : B[j-1]);
		  int Ai   = ((i >= m) ? Integer.MAX_VALUE : A[i]);
		  int Bj   = ((j >= n) ? Integer.MAX_VALUE : B[j]);
		 
		  if (Bj_1 < Ai && Ai < Bj)
		    return Ai;
		  else if (Ai_1 < Bj && Bj < Ai)
		    return Bj;
		 
		  assert((Ai > Bj && Ai_1 > Bj) || 
		         (Ai < Bj && Ai < Bj_1));
		 
		  // if none of the cases above, then it is either:
		  if (Ai < Bj)
		    // exclude Ai and below portion
		    // exclude Bj and above portion
		    return findKthSmallest(A, i+1, r2, m-i-1, B, p2, j-1, j, k-i-1);
		  else /* Bj < Ai */
		    // exclude Ai and above portion
		    // exclude Bj and below portion
		    return findKthSmallest(A, p2, i-1, i, B, j+1, r2, n-j-1, k-j-1);
		}
	
	public static double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if(nums1.length > nums2.length){
    		return mediaTwoSortedArrays(nums2, 0, nums2.length-1, nums1, 0, nums1.length-1);
    	}
    	else{
            return mediaTwoSortedArrays(nums1, 0, nums1.length-1, nums2, 0, nums2.length-1);
    	}
    }
    public static double mediaTwoSortedArrays(int[] A, int p1, int r1, int[] B, int p2, int r2){
    	int n1 = r1-p1+1;
    	int n2 = r2-p2+1;
    	
    	if(n1 > n2){
    		return mediaTwoSortedArrays(B, p2, r2, A, p1, r1);
    	}
    	
    	n1 = r1-p1+1;
    	n2 = r2-p2+1;
    	
    	if(n1 <= 0){
			if(n2%2==1){
				return B[n2/2];
			}
			else{
				return (B[n2/2-1]+B[n2/2])/2.0;
			}
		}
    	
    	//base cases
    	//n1=1
    	if(n1 == 1){
    		//n2=1
    		if(n2 == 1){
    			return (A[0]+B[0])/2.0;
    		}
    		//if n2 is odd - then A[0] potentially be either left or right of B[n2/2]. In that case 
    		//median is the avaerage of B[n2/2] and median of numbers A[0], B[n2/2-1], and B[n2/2+1]
    		if(n2%2 == 1){
    			return (B[n2/2] + (A[0]+B[n2/2-1]+B[n2/2+1]- Math.max(A[0], Math.max(B[n2/2-1], B[n2/2-1]))) - Math.min(A[0], Math.min(B[n2/2-1], B[n2/2-1])))/2.0;
    		}
    		//if n2 is even then median is the median of the 2 middle numbers of B and A[0]
    		else{
    			return (A[0]+B[n2/2]+B[n2/2-1])- Math.max(A[0], Math.max(B[n2/2], B[n2/2-1])) - Math.min(A[0], Math.min(B[n2/2], B[n2/2-1]));
    		}
    	}
    	//n1 =2 
    	else if(n1 == 2){
    		//n2 = 2
    		if(n2 == 2){
    			int max = Math.max(Math.max(A[0], A[1]), Math.max(B[0], B[1]));
    			int min = Math.min(Math.min(A[0], A[1]), Math.min(B[0], B[1]));
    			
    			return (A[0]+A[1]+B[0]+B[1]-max-min)/2;
    		}
    		//if n2 is odd then media will be one of B[n2/2], max of A[0] or B[n2/2-1], and min of A[1] and B[n/2+1]
    		if(n2%2 == 1){
//    			 if(n2 == 1){
//     		        return (B[0]+A[0]+A[1])-Math.max(B[0], Math.max(A[0], A[1])) - Math.min(B[0], Math.min(A[0], A[1]));
//     		    }
    			 
    			int maxLeft = Math.max(A[0], B[n2/2-1]);
    			int minRight = Math.min(A[1], B[n2/2+1]);
    			
    			return (B[n2/2]+maxLeft+minRight)-Math.max(B[n2/2], Math.max(maxLeft, minRight)) - Math.min(B[n2/2], Math.min(maxLeft, minRight));
    		}
    		//if n2 is even - median is among middle two elements of B and 2 elements on the left and right 
    		// which can be replaced by two elements of A
    		else{
    			int maxLeft = Math.max(A[0], B[n2/2-2]);
    			int minRight = Math.min(A[1], B[n2/2+1]);
    			
    			int max = Math.max(Math.max(B[n2/2], B[n2/2-1]), Math.max(maxLeft, minRight));
    			int min = Math.min(Math.min(B[n2/2], B[n2/2-1]), Math.min(maxLeft, minRight));
    			
    			return (B[n2/2]+B[n2/2-1]+maxLeft+minRight-max-min)/2.0;
    		}
    	}
    	
    	int mid1 = (n1)/2;
    	int mid2 = (n2)/2;
    	
    	//divide and conquer
    	//mid1 is less then mid2 -  so we median exists in A[mid1+1...] and B[...mid2]
    	if(A[mid1] <= B[mid2]){
    		return mediaTwoSortedArrays(A, mid1+1, r1, B, p2, mid2-1);
    	}
    	else{
    		return mediaTwoSortedArrays(A, p1, mid1-1, B, mid2+1, r2);
    	}
    }
   
    //container/tower with most water
	public static int maxArea(int[] height) {
		int len = height.length, low = 0, high = len - 1;
		int maxArea = 0;
		while (low < high) {
			maxArea = Math.max(maxArea,
					(high - low) * Math.min(height[low], height[high]));
			if (height[low] < height[high]) {
				low++;
			} else {
				high--;
			}
		}
		return maxArea;
	}
	
	public static void DNFSort(int a[]){
		int p1 = 0;
		int p2 = 0;
		int p3 = a.length-1;
		
		while(p2 <= p3){
			if(a[p2] == 0){
				swap(a, p2, p1);
				p1++;
				p2++;
			}
			else if(a[p2] == 1){
				p2++;
			}
			else if(a[p2] == 2){
				swap(a, p2, p3);
				p3--;
			}
		}
	}
	
	public static int activitySelecyion(Job[] pairs){
		Arrays.sort(pairs); //Assume that Pair class implements comparable with the compareTo() method such that (a, b) < (c,d) iff b<c
		int chainLength = 0;
		
		//select the first pair of the sorted pairs array
		pairs[0].print(); //assume print method prints the pair as “(a, b) ”
		chainLength++;
		int prev = 0;

		for(int i=1;i<pairs.length; i++)
		{
			if(pairs[i].start >= pairs[prev].finish)
			{
				chainLength++;
				prev = i;
			}
		}
		return chainLength;	
	}
	
	//largest element smaller than or equal to key
	public static int binarySearchFloor(int A[], int l, int h, int key){
		int mid = (l+h)/2;
		
		if(A[h] <= key){
			return h;
		}
		if(A[l] > key ){
			return -1;
		}
		
		if(A[mid] == key){
			return mid;
		}
		//mid is greater than key, so floor is either mid-1 or it exists in A[l..mid-1]
		else if(A[mid] > key){
			if(mid-1 >= l && A[mid-1] <= key){
				return mid-1;
			}
			else{
				return binarySearchFloor(A, l, mid-1, key);
			}
		}
		//mid is less than key, so floor is either mid or it exists in A[mid+1....h]
		else{
			if(mid+1 <= h && A[mid+1] > key){
				return mid;
			}
			else{
				return binarySearchFloor(A, mid+1, h, key);
			}
		}
	}
	
	//smallest element greater than or equal to key
	public static int binarySearchCeiling(int A[], int l, int h, int key){
		int mid = (l+h)/2;
		
		if(A[l] >= key){
			return l;
		}
		if(A[h] < key ){
			return -1;
		}
		
		if(A[mid] == key){
			return mid;
		}
		//mid is greater than key, so either mid is the ceil or it exists in A[l..mid-1]
		else if(A[mid] > key){
			if(mid-1 >= l && A[mid-1] <= key){
				return mid;
			}
			else{
				return binarySearchCeiling(A, l, mid-1, key);
			}
		}
		//mid is less than the key, so either mid+1 is the ceil or it exists in A[mid+1...h]
		else{
			if(mid + 1 <= h && A[mid+1] >= key){
				return mid+1;
			}
			else{
				return binarySearchCeiling(A, mid+1, h, key);
			}
		}
	}
	
	//largest element smaller than or equal to key
	public static int binarySearchLatestCompatibleJob(Job[] A, int l, int h, int key){
		int mid = (l+h)/2;
		
		if(A[h].finish <= key){
			return h;
		}
		if(A[l].finish > key ){
			return -1;
		}
		
		if(A[mid].finish == key){
			return mid;
		}
		//mid is greater than key, so floor is either mid-1 or it exists in A[l..mid-1]
		else if(A[mid].finish > key){
			if(mid-1 >= l && A[mid-1].finish <= key){
				return mid-1;
			}
			else{
				return binarySearchLatestCompatibleJob(A, l, mid-1, key);
			}
		}
		//mid is less than key, so floor is either mid or it exists in A[mid+1....h]
		else{
			if(mid+1 <= h && A[mid+1].finish > key){
				return mid;
			}
			else{
				return binarySearchLatestCompatibleJob(A, mid+1, h, key);
			}
		}
	}
	
	public static int weightedActivitySelection(Job[] jobs){
		int n = jobs.length;
		int profit[] = new int[n+1];
		int q[] = new int[n];
		
		//sort according to finish time
		Arrays.sort(jobs);
		
		//find q's - O(nlgn)
		for(int i = 0; i< n; i++){
			q[i] = binarySearchLatestCompatibleJob(jobs, 0, n-1, jobs[i].start);
		}
		
		//compute optimal profits - O(n)
		profit[0] = 0;
		for(int j = 1; j<=n; j++){
			int profitExcluding = profit[j-1];
			int profitIncluding = jobs[j-1].weight;
			if(q[j-1] != -1){
				profitIncluding += profit[q[j-1]+1];
			}
			profit[j] = Math.max(profitIncluding, profitExcluding);
		}
		return profit[n];
	}
	
	public static int maxOverlapIntervalCount(int[] start, int[] end){
		int maxOverlap = 0;
		int currentOverlap = 0;
		
		Arrays.sort(start);
		Arrays.sort(end);
		
		int i = 0;
		int j = 0;
		int m=start.length,n=end.length;
		while(i< m && j < n){
			if(start[i] < end[j]){
				currentOverlap++;
				maxOverlap = Math.max(maxOverlap, currentOverlap);
				i++;
			}
			else{
				currentOverlap--;
				j++;
			}
		}
		
		return maxOverlap;
	}
	
	private static int longestPalindromLength(String str, int l, int r, int n){
		int len = 0;
		while(l >= 0 && r <= n-1 && str.charAt(l--) == str.charAt(r++)){
			len++;
		}
		
		return len;
	}
	public static String longestPalindrom(String str){
		int n = str.length();
		if(n <= 1){
			return str;
		}
		
		int l = 0;
		int h = 0;
		int start = 0;
		int maxlen = 1;
		
		for(int i = 1; i < n; i++){
			//palindrom of even length with centers i-1 and i
			l = i-1;
			h = i;
			int len = 0;
			while(l >= 0 && h <= n-1 && (str.charAt(l--) == str.charAt(h++))){
				len = h-l+1;
				if(len > maxlen){
					start = l;
					maxlen = len;
				}
				l--;
				h++;
			}
			
			//palindrom of odd length with center at i
			l = i;
			h = i;
			while(l >= 0 && h <= n-1 && (str.charAt(l) == str.charAt(h))){
				len = h-l+1;
				if(len > maxlen){
					start = l;
					maxlen = len;
				}
				l--;
				h++;
			}
		}
		
		return str.substring(start, start+maxlen);
	}
	
	public static int longestPalindromDP(String str){
		int n = str.length();
		int dp[][] = new int[n+1][n+1];
		for(int i = 1; i<n; i++){
			dp[i][i] = 1;
		}
		
		//find palindrom of each possible length
		for(int len = 2; len <= n; len++){
			//try to get a palindrom of length len starting at each position i
			for(int i = 1; i <= n-len+1; i++){
				//right end position of current palindrom
				int j = i+len-1;
				
				if(str.charAt(i-1) == str.charAt(j-1)){
					dp[i][j] = 2+dp[i+1][j-1];
				}
				else{
					dp[i][j] = Math.max(dp[i][j-1], dp[i+1][j]);
				}
			}
		}
		
		return dp[1][n];
	}
	
	private static boolean equalHistogram(int[] hist1, int[] hist2){
		for(int i = 0; i < hist1.length; i++){
			if(hist1[i] != hist2[i]){
				return false;
			}
		}
		
		return true;
	}
	
	public static int searchAnagramSubstring(String text, String pattern){
		int count = 0;
		int n = text.length();
		int m = pattern.length();
		
		if(n < m | m == 0 | m == 0){
			return 0;
		}
			
			
		int textHist[] = new int[256];
		int patHist[] = new int[256];
		
		//initial histogram window of size m 
		int i = 0;
		for(i = 0; i < m; i++){
			patHist[pattern.charAt(i)]++;
			textHist[text.charAt(i)]++;
		}
	
		//search the pattern histogram in a sliding window of text histogram
		do{
			//O(1) time check as array size is constant
			if(equalHistogram(textHist, patHist)){
				System.out.println("anagram found : "+text.substring(i-m, i));
				count++;
			}
			
			//slide the text histogram window by 1 position to the right and check for anagram
			textHist[text.charAt(i)]++;
			textHist[text.charAt(i-m)]--;
		} while(++i < n);
		
		return count;
	}
	
	
	public static class Dictionary{
		private Set<String> dictionary = new HashSet<String>();
		
		public void add(String word){
			dictionary.add(word);
		}
		public void addAll(List<String> words){
			dictionary.addAll(words);
		}
		public boolean remove(String word){
			return dictionary.remove(word);
		}
		private String getKey(String str){
			str = str.toLowerCase().trim();
			int[] hist = new int[256];
			for(int i = 0; i < str.length(); i++){
				hist[str.charAt(i)]++;
			}
			
			StringBuilder sb = new StringBuilder();
			
			for(int val : hist){
				sb.append(val);
			}
			
			return sb.toString();
		}
		public int searchAnagram(String pattern){
			int count = 0;
			HashMap<String, List<String>> histogramMap = new HashMap<String, List<String>>();
			
			for(String word : dictionary){
				String key = getKey(word);
				
				if(!histogramMap.containsKey(key)){
					histogramMap.put(key, new ArrayList<String>());
				}
				
				histogramMap.get(key).add(word);
			}
			
			String searchKey = getKey(pattern);
			List<String> res = histogramMap.get(searchKey);
			
			if(res != null){
				count = res.size();
				
				System.out.print("anagrams in dict: ");
				for(String s : res){
					System.out.print(s+" ");
				}
				System.out.println();
			}
			
			return count;
		}
	}
	
	public static int[] slidingWindowMax(final int[] in, final int w) {
	    final int[] max_left = new int[in.length];
	    final int[] max_right = new int[in.length];
	
	    max_left[0] = in[0];
	    max_right[in.length - 1] = in[in.length - 1];
	
	    for (int i = 1; i < in.length; i++) {
	        max_left[i] = (i % w == 0) ? in[i] : Math.max(max_left[i - 1], in[i]);
	
	        final int j = in.length - i - 1;
	        max_right[j] = (j % w == 0) ? in[j] : Math.max(max_right[j + 1], in[j]);
	    }
	
	    final int[] sliding_max = new int[in.length - w + 1];
	    for (int i = 0, j = 0; i + w <= in.length; i++) {
	        sliding_max[j++] = Math.max(max_right[i], max_left[i + w - 1]);
	    }
	
	    return sliding_max;
	}
	
	public static int[] slidingWindowMin(final int[] in, final int w) {
	    final int[] min_left = new int[in.length];
	    final int[] min_right = new int[in.length];
	
	    min_left[0] = in[0];
	    min_right[in.length - 1] = in[in.length - 1];
	
	    for (int i = 1; i < in.length; i++) {
	        min_left[i] = (i % w == 0) ? in[i] : Math.min(min_left[i - 1], in[i]);
	
	        final int j = in.length - i - 1;
	        min_right[j] = (j % w == 0) ? in[j] : Math.min(min_right[j + 1], in[j]);
	    }
	
	    final int[] sliding_max = new int[in.length - w + 1];
	    for (int i = 0, j = 0; i + w <= in.length; i++) {
	        sliding_max[j++] = Math.min(min_right[i], min_left[i + w - 1]);
	    }
	
	    return sliding_max;
	}
	
	public static int[] smallerCountOnRight(final int[] X) {
		int[] smaller = new int[X.length];
	    for(int i = 0; i < X.length; i++){
	    	for(int j = i+1; j < X.length; j++){
	    		if(X[j] <= X[i]){
	    			smaller[i]++;	
	    		}
	    	}
	    }
	    
	    return smaller;
	}
	
	public static int permRank(final int[] X) {
	    final int[] smaller_count = countSmallerOnRightWithMerge(X);
	
	    final int[] factorial = new int[X.length];
	    factorial[0] = 1;
	    factorial[1] = 1;
	
	    for (int i = 2; i < X.length; i++) {
	        factorial[i] = i*factorial[i - 1];
	    }
	
	    int rank = 1;
	    for (int i = 0; i < X.length; i++) {
	        rank += smaller_count[i] * factorial[X.length - i - 1];
	    }
	
	    return rank;
	}
	
	public static char firstUnique(char[] stream){
		HashSet<Character> seen = new HashSet<Character>();
		LinkedHashSet<Character> uniques = new LinkedHashSet<Character>();
		
		for(int i = 0; i < stream.length; i++){
			char c = Character.toLowerCase(stream[i]);
			if(!seen.contains(c)){
				seen.add(c);
				uniques.add(c);
			}
			else{
				uniques.remove(c);
			}
		}
		
		if(uniques.size() > 0){
			return uniques.iterator().next();
		}
		else return '\0';
	}
	
	public static class BTNode{
		int key;
		int val;
		BTNode left;
		BTNode right;
		BTNode next;
		
		public BTNode(int key){
			this.key = key;
			left = null;
			right = null;
		}
	
		@Override
		public String toString() {
			return key+"";
		}
		
		public void print(BTNode root){
			if(root == null){
				return;
			}
			
			print(root.left);
			System.out.print(root.key+", ");
			print(root.right);
		}
	}
	
	 public static void paths(BTNode root, String path, ArrayList<String> paths){
		 if(root == null){
			 return;
		 }
		 
		 path=path+(path.isEmpty()? "" : "-->")+root.key;
		 
		 if(root.left == null && root.right == null){
			 System.out.println("path > "+path);
			 paths.add(path);
			 return;
		 }
		 
		 paths(root.left, path, paths);
		 paths(root.right, path, paths);
	 }
	 
	 public static int maxSumPath1(BTNode node, int[] maxSum){
		 if(node ==  null){
			 return 0;
		 }
		 
		 int leftSum = maxSumPath(node.left, maxSum);
		 int rightSum = maxSumPath(node.right, maxSum);
		 
		 //update global max so far
		 //we take max by either going through current root or not going through
		 //current root if we take root then the max sum path goes from a left 
		 //subtree node through root to a right subtree node
		 //if we don't take current root then we disregard path 
		 //through root as if we have max path in either left or right subtree
		 maxSum[0] = Math.max(maxSum[0], leftSum+rightSum+node.key);
		 
		 //return max in this current path that starts or ends at current root and doesn't go through it
		 return Math.max(leftSum, rightSum) + node.key;
	 }
	 
	 public static int maxSumPath(BTNode node, int[] maxSum){
		 if(node ==  null){
			 return 0;
		 }
		 
		 int leftSum = maxSumPath(node.left, maxSum);
		 int rightSum = maxSumPath(node.right, maxSum);
		 
		 //update global max so far
		 //we take max by either taking current root or not taking current root
		 //if we take root then the max sum path goes from a left 
		 //subtree node through root to a right subtree node
		 //if we don't take current root then we disregard path 
		 //through root as if we have max path in either left or right subtree
		 int localMaxSum = Math.max(Math.max(leftSum, rightSum) + node.key, node.key);
		 //global max
		 int globalMax = Math.max(localMaxSum, leftSum+rightSum+node.key);
		 //maxSum[0] = Math.max(maxSum[0], leftSum+rightSum+node.key);
		 //update global max
		 maxSum[0] = Math.max(maxSum[0], globalMax);
		 
		 //return max in this current path that starts or ends at current root and doesn't go through it
		 //return Math.max(leftSum, rightSum) + node.key;
		 return localMaxSum;
	 }
	 
	 public static int getLevel(BTNode root, int count, BTNode node){
		 if(root == null){
			 return 0;
		 }
		 
		 if(root == node){
			 return count;
		 }
		 
		 int leftLevel = getLevel(root.left, count+1, node);
		 if(leftLevel != 0){
			 return leftLevel;
		 }
		 int rightLevel = getLevel(root.right, count+1, node);
		 return rightLevel;
	 }
	 
	 public static BTNode LCA(BTNode root, BTNode x, BTNode y) {
		  if (root == null) return null;
		  if (root == x || root == y) return root;
	
		  BTNode leftSubTree = LCA(root.left, x, y);
		  BTNode rightSubTree = LCA(root.right, x, y);
	
		  //x is in one subtree and and y is on other subtree of root
		  if (leftSubTree != null && rightSubTree != null) return root;  
		  //either x or y is present in one of the subtrees of root or none present in either side of the root
		  return leftSubTree!=null ? leftSubTree : rightSubTree;  
	}
	 
	 public static int shortestDistance(BTNode root, BTNode a, BTNode b){
		 if(root == null){
			 return 0;
		 }
		 
		 BTNode lca = LCA(root, a, b);
		 //d(a,b) = d(root,a) + d(root, b) - 2*d(root, lca)
		 return getLevel(root, 1, a)+getLevel(root, 1, b)-2*getLevel(root, 1, lca);
	 }
	 
	 public static int maxDepth(BTNode root) {
	     
	     if(root == null){
	         return 0;
	     }
	     
	     int leftDepth = maxDepth(root.left);
	     int rightDepth = maxDepth(root.right);
	     
	     return Math.max(leftDepth, rightDepth)+1;     
	 }
	 
	public static void allCasePermutataion(String str, int start, Set<String> res){
		if(start == str.length()){
			res.add(str);
			return;
		}
		
		char[] chars = str.toCharArray();
		chars[start] = Character.toLowerCase(chars[start]);
		allCasePermutataion(new String(chars), start+1, res);
		chars[start] = Character.toUpperCase(chars[start]);
		allCasePermutataion(new String(chars), start+1, res);
	}
	
	public static int diameter(BTNode root){
		//D(T) = max{D(T.left), D(T.right), LongestPathThrough(T.root)}
		if(root == null){
			return 0;
		}
		
		int leftHeight = maxDepth(root.left);
		int rightHeight = maxDepth(root.right);
		
		int leftDiameter = diameter(root.left);
		int rightDiameter = diameter(root.right);
		
		return Math.max(Math.max(leftDiameter, rightDiameter), leftHeight+rightHeight+1);
	}
	
	public static int diameter(BTNode root, int[] height){
		if(root == null){
			height[0] = 0;
			return 0;
		}
		
		int[] leftHeight = {0}, rightHeight = {0};
		int leftDiam = diameter(root.left, leftHeight);
		int rightDiam = diameter(root.right, rightHeight);
		
		height[0] = Math.max(leftHeight[0],rightHeight[0])+1;
		
		return Math.max(Math.max(leftDiam, rightDiam), leftHeight[0]+rightHeight[0]+1);
	}
	
	public static void mirrorTree(BTNode root){
		if(root == null){
			return;
		}
		
		mirrorTree(root.left);
		mirrorTree(root.right);
		
		BTNode temp = root.right;
		root.right = root.left;
		root.left = temp;
	}
	
	public static ListNode splitLinkedListNode2(ListNode head, int n){
		ListNode slow = head;
		ListNode fast = head;
		ListNode prev = head;
		
		while(fast != null && slow != null){
			int count = 0;
			prev = slow;
			slow=slow.next;
			while(count < n && fast != null){
				fast = fast.next;
				count++;
			}
			
			if(slow == fast){
				return null;
			}
		}
		
		if(prev != null){
			prev.next = null;
		}
		
		return slow;
	}
	
	public static boolean detectCycle(ListNode head){
	    ListNode slow = head;
	    ListNode fast = head.next;
	    
	    while(slow != null && fast != null && slow != fast){
	        if(fast.next == null){
	             break;
	        }
	        slow = slow.next;
	        fast = fast.next.next;
	    }
	    
	    if(slow != null && fast != null && slow == fast){
	        return true;
	    }
	    
	    return false;
	}
	public static int size(ListNode head){
	    ListNode temp = head;
	    int count  = 0;
	    
	    while(temp != null){            
	        count++;
	        temp = temp.next;
	    }
	    
	    return count;
	}   
	
	public static ListNode splitLinkedListNode(ListNode head, int n){
		//O(size)
	    if(head == null || detectCycle(head)){
	       return null;
	    }
	    //O(size)
	    int size = size(head);
	
	    if(n >= size || size == 0){
	        return null;
	    }
	    
	    int split = 0;
	    if(size % n == 0){
	        split = size/n;
	    }
	    else{
	        split = (size-1)/n+1;
	    }
	    
	    //O(n)
	    ListNode temp = head;
	    ListNode last = head;
	    while(split > 0){
	        last = temp;
	        temp = temp.next;
	        split--;
	    }
	    
	    last.next = null;
	    
	    return temp;
	}
	
	public static void reverse(int A[], int i, int j){
		while(i < j){
			swap(A, i, j);
			i++;
			j--;
		}
	}
	
	public static void nextPermutation(int[] nums) {
        int k = -1;
        for (int i = nums.length - 2; i >= 0; i--) {
            if (nums[i] < nums[i + 1]) {
                k = i;
                break;
            }
        } 
        if (k == -1) {
            reverse(nums, 0, nums.length-1);
            return;
        }
        int l = -1;
        for (int i = nums.length - 1; i > k; i--) {
            if (nums[i] > nums[k]) {
                l = i;
                break;
            } 
        } 
        swap(nums, k, l);
        reverse(nums, k + 1, nums.length-1); 
    }
	
	public static void mergeSortedAlternate(int[] a, int[] b, ArrayList<Integer> c, int[] index){
		int i = index[0];
		int j = index[1];
		
		for(int k = i; k < a.length;){
			if(j >= b.length){
				if(c.size() % 2 == 1){
					c.add(Math.max(a[i], b[j-1]));
					System.out.println(Arrays.toString(c.toArray()));
				}
				
				return;
			}
			if(a[k] > b[j]){
				j++;
			}
			else{
				if(j < b.length){
					c.add(a[k]);
					
					if(c.size() % 2 == 0){
						System.out.println(Arrays.toString(c.toArray()));
					}
				}
				
				k++;
				index[0] = j;
				index[1] = k;
				mergeSortedAlternate(b, a, c, index);
				c = new ArrayList<Integer>();
			}
		}
	}
	
	public static ListNode reverse(ListNode head){
		if(head == null || head.next == null){
			return head;
		}
		
		ListNode reversed = null;
		ListNode temp = null;
		while(head != null){
			temp = head.next;
			head.next = reversed;
			reversed = head;
			head = temp;
		}
		
		return reversed;
	}
	
	//recursive: Idea is to take each element from the list and add in front of another list called reversed
	public static ListNode reverse(ListNode head, ListNode reversed){
		if(head==null) return reversed;
		ListNode current = new ListNode(-1);
		current = head;
		head = head.next;
		current.next = reversed;
		return reverse(head, current);
	}
	
	public static ListNode reverse(ListNode head, int k){
		if(head == null || head.next == null){
			return head;
		}
		
		ListNode prevHead = head;
		ListNode reversed = null;
		ListNode temp = null;
		int count = 0;
		while(head != null && count < k){
			temp = head.next;
			head.next = reversed;
			reversed = head;
			head = temp;
			count++;
		}
		
		if(prevHead != null){
			prevHead.next = reverse(head, k);
		}
		
		return reversed;
	}
	
	public static ListNode deleteNodeWithHighOnRight(ListNode head){
		ListNode temp = null;
		ListNode newHead = null;
		ListNode prev = head;
		while(head != null && head.next != null){
			temp = head.next;
			
			if(temp.val > head.val){
				prev.next = temp;
			}
			else {
				if(newHead == null){
					newHead = head;
				}
				prev = head;
			}
			
			head = head.next;
		}
		
		return newHead;
	}
	
	public void removeCycle(ListNode head){
		ListNode slow = head;
		ListNode fast = head.next;
		
		while(fast != null && fast.next != null){
			if(slow == fast){
				break;
			}
			slow = slow.next;
			fast = fast.next.next;
		}
		
		if(slow == fast){
			slow = head;
			while(slow != fast.next){
				slow = slow.next;
				fast = fast.next;
			}
			
			fast.next = null;
		}
	}
	
	public static ListNode mergeSortedLists(ListNode a, ListNode b){
		if(a == null){
			return b;
		}
		if(b == null){
			return a;
		}
		
		ListNode merged = null;
		
		if(a.val > b.val){
			merged = b;
			merged.next = mergeSortedLists(a, b.next);
		}
		else{
			merged = a;
			merged.next = mergeSortedLists(a.next, b);
		}
		
		return merged;
	}
	
	public static void MergeSortList(ListNode head){
		if(head != null && head.next != null){
			ListNode left = head;
			ListNode right = splitLinkedListNode2(left, 2);
			MergeSortList(left);
			MergeSortList(right);
			
			head = mergeSortedLists(left, right);
		}
	}
	
	public static void printLevelOrder(BTNode root){
		Queue<BTNode> queue = new LinkedList<test.BTNode>();
		queue.offer(root);
		
		BTNode node = null;
		int count = 1;
		while(!queue.isEmpty()){
			node = queue.poll();
			count--;
			
			System.out.print(node.key+" ");
			if(node.left != null){
				queue.offer(node.left);
			}
			if(node.right != null){
				queue.offer(node.right);
			}
			
			if(count == 0){
				System.out.println("");
				count = queue.size();
			}
		}
	}
	
	public static void connectLevelOrder(BTNode root){
		Queue<BTNode> queue = new LinkedList<test.BTNode>();
		queue.offer(root);
		
		BTNode node = null;
		int count = 1;
		while(!queue.isEmpty()){
			if(node == null){
				node = queue.poll();
				node.next = null;
			}
			else{
				node.next = queue.poll();
				node = node.next;
			}
			count--;
			
			System.out.print(node.key+" ");
			if(node.left != null){
				queue.offer(node.left);
			}
			if(node.right != null){
				queue.offer(node.right);
			}
			
			if(count == 0){
				node = null;
				System.out.println("");
				count = queue.size();
			}
		}
	}
	
	public static int minInsertionsForLongestPalindrom(final String str) {
	    final int n = str.length();
	    // L[i][j] contains minimum number of deletes required to make string(i..j) a palindrome
	    final int[][] L = new int[n][n];

	    // find L for each pair of increasing range (i,j) where i<=j. That is we are only populating upperhalf of the
	    // table
	    for (int i = 1; i < n; i++) {
	        for (int j = i, k = 0; j < n; j++, k++) {
	            // if characters are same at the two boundary then no deletions required in these positions.
	            // if characters are not same the we can insert either string[i] at the gap between j-1 and j or we
	            // can insert string[j] at the gap between i and i+1. We take the min of these
	            L[k][j] = str.charAt(k) == str.charAt(j) ? L[k + 1][j - 1] : Math.min(L[k][j - 1], L[k + 1][j]) + 1;
	        }
	    }

	    return L[0][n - 1];
	}
	
	public static int numOfUniqueBSTDP(int n, int[] counts){
		int count = 0;
		if(n < 0){
			return 0;
		}
		if(n <= 1){
			return 1;
		}
		
		//count possible trees with each element as root
		for(int i = 1; i<=n; i++){
			//compute if not in DP
			if(counts[i] == -1){
				counts[i] = numOfUniqueBSTDP(i-1, counts);
			}
			if(counts[n-i] == -1){
				counts[n-i] = numOfUniqueBSTDP(n-i, counts);
			}
			
			count +=  counts[i-1]+counts[n-i];
		}
		
		counts[n] = count;
		return count;
	}
	
	public static int numOfUniqueBSTDP(int m, int n){
		int len = n-m+1;
		int[] counts = new int[n+1]; 
		
		//mark each cell not computed
		for(int i = 0; i<=n; i++){
			counts[i] = -1;
		}
		
		return numOfUniqueBSTDP(len, counts);
	}
	
	public static int numOfUniqueBST1(int len){
	    if(len <= 1){
	      return 1;
	    }
	    else{
	      int count = 0;
	      for(int i = 1; i<= len; i++){
	       count += numOfUniqueBST1(i-1)*numOfUniqueBST1(len-i);
	      }   
	      return count;
	    }
	  }
	  
	  public static int numOfUniqueBST1(int m, int n) {
	    int len = n-m+1;
	    return numOfUniqueBST1(len);
	  }
	  
	  public static void printIPAddressedd(String file){
		    //read file line by line
		  	String[] lines = {"Oct 16 03:18:05 app1002.corp httpd: 172.18.159.102 - - [16/Oct/2013:03:18:01 +0000] \"GET / HTTP/1.1\" 200 22 \"-\" \"Python-CRT-Client/0.0.8\" 3378887", "Oct 16 03:18:05 web1004.corp httpd: 202.16.73.36 - - [16/Oct/2013:03:18:05 +0000] \"GET /icon.gif HTTP/1.1\" 404 310 \"-\" \"Python-urllib/2.6\" 200", "Dec 16 05:04:45 mail3.corp postfix/smtpd[26986]: disconnect from 172.16.73.2", "Dec 16 05:04:45 app1003.corp postfix/smtpd[26965]: client=172.32.72.5"};
		    String validIp = "(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)";
		    String ipAddressPatternRegex = validIp+"\\."+validIp+"\\."+validIp+"\\."+validIp;
		    Matcher m;
		    HashSet<String> ips = new HashSet<String>();
		    for(String line : lines){
		        Pattern p = Pattern.compile(ipAddressPatternRegex);
		        m = p.matcher(line);
		        while(m.find()){
		            String ip = m.group().trim();
		            if(!ips.contains(ip)){
		                ips.add(ip);
		                System.out.println(ip);
		            }
		        }
		    }        
		}
	  
	  public static int maxSumSubArray(final int[] a) {
		    int maxSum = a[0];
		    int maxSoFar = a[0];
		    int tempBegin = 0;
		    int begin = 0;
		    int end = 0;
	
		    for (int i = 1; i < a.length; i++) {
		        maxSoFar += a[i];
	
		        if (maxSoFar < 0) {
		            maxSoFar = 0;
		            tempBegin = i + 1;
		        } else if (maxSoFar > maxSum) {
		            maxSum = maxSoFar;
		            begin = tempBegin;
		            end = i;
		        }
		    }
	
		    for (int i = begin; i <= end; i++) {
		        System.out.print(a[i] + ", ");
	    }
	    return maxSum;
	}
	  
	public static ListNode flattenBT2SLL(BTNode root){
		if(root == null){
			return null;
		}
		
		ListNode head = new ListNode(root.key);
		ListNode left = null;
		ListNode right = null;
		if(root.left != null){
			left = flattenBT2SLL(root.left);
		}
		if(root.right != null){
			right = flattenBT2SLL(root.right);
		}
		
		head.next = right;
		ListNode predecessor = left;
		while(predecessor != null && predecessor.next != null){
			predecessor =  predecessor.next;
		}
		
		if(predecessor != null){
			predecessor.next = head;
			head = left;
		}
		
		return head;
	}
	
	public static void flattenBT2SLLFaster(BTNode root, LinkedList<BTNode> head){
		if(root == null){
			return ;
		}
		
		flattenBT2SLLFaster(root.left, head);
		head.add(root);
		flattenBT2SLLFaster(root.right, head);
	}
	
	//The idea of inplace is to use tree's left and right pointer to connect linked list nodes
	//It is possible because once we visit childs of a node we don't actually need left/right 
	//pointers anymore. So, we can reuse them for pointing prev/next nodes.
	public static void flattenBT2LLPreOrderInplace(BTNode root){
		if(root == null){
			return;
		}
		
		BTNode cur = root;
		while(cur != null){
			//if cur has a left child then we would like to flatten the left subtree recursively
			//and put then under right child of cur so that we get a flatten list by right pointer to traverse
			//We put left subtree first and then right subtree
			if(cur.left != null){
				//As we will put flattened left subtree to the right pointer of cur so
				//before doing that we need to point the last (rightmost) node of flattened left subtree
				//to point to right subtree (if it exists)
				if(cur.right != null){
					BTNode last = cur.left;
					while(last.right != null){
						last = last.right;
					}
					
					last.right = cur.right;
				}
				
				//now update next (right) pointer of cur node to flattened left subtree
				cur.right = cur.left;
				cur.left = null;//Single Linked list - so no prev pointer
			}
			//if thers is no left subtree the we directly go to right subtree and flatten it out
			else{
				cur = cur.right;
			}
		}
	}
	
	public static BTNode convertList2BTreeRecursive(ListNode head){
		int n = 0;
		ListNode temp = head;
		while(temp != null){
			n++;
			temp = temp.next;
		}
		
		return convertList2BTreeRecursive(head, 0, n-1);
	}
	
	//works for sorted/unsorted single/double linked list and for both BT and BST
	public static BTNode convertList2BTreeRecursive(ListNode h, int start, int end){
		if(start > end){
			return null;
		}
		
		//keep halving
		int mid = (start)+(end-start)/2;
		
		//build left subtree
		BTNode left = convertList2BTreeRecursive(h, start, mid-1);
		//build root from current node
		BTNode root = new BTNode(h.val);
		//update left
		root.left = left;
		//build right subtree - first we need to increment head pointer 
		//java pass objects by reference , so we can't just do h = h.next
		//instead we can update the head by value of head.next
		//head = head.next;
		if(h.next != null){
			h.val = h.next.val;
			h.next = h.next.next;
			root.right = convertList2BTreeRecursive(h, mid+1, end);
		}
		
		return root;
	}
	
	public static BTNode convertList2BTreeIterative(ListNode head){
		if(head == null){
			return null;
		}
		Queue<BTNode> queue = new LinkedList<BTNode>();
		BTNode root = new BTNode(head.val);
		head = head.next;
		queue.offer(root);
		
		BTNode node = null;
		while(!queue.isEmpty()){
			node = queue.poll();
			if(head != null){
				node.left = new BTNode(head.val);
				head = head.next;
				queue.offer(node.left);
			}
			if(head != null){
				node.right = new BTNode(head.val);
				head = head.next;
				queue.offer(node.right);
			}
		}
		
		return root;
	}
	
	public static ListNode convertList2BTreeInplace(ListNode head){
		int n = 0;
		ListNode temp = head;
		while(temp != null){
			n++;
			temp = temp.next;
		}
		
		return convertList2BTreeInplace(head, 0, n-1);
	}
	
	//works in place for sorted/unsorted single/double linked list and for both BT and BST
	public static ListNode convertList2BTreeInplace(ListNode h, int start, int end){
		if(start > end){
			return null;
		}
		
		//keep halving
		int mid = (start)+(end-start)/2;
		
		//build left subtree
		ListNode left = convertList2BTreeInplace(h, start, mid-1);
		//build root from current node
		ListNode root = new ListNode(h.val);//h;
		//update left
		root.prev = left;
		//build right subtree - first we need to increment head pointer 
		//java pass objects by reference , so we can't just do h = h.next
		//instead we can update the head by value of head.next
		//head = head.next;
		if(h.next != null){
			h.val = h.next.val;
			h.next = h.next.next;
			root.next = convertList2BTreeInplace(h, mid+1, end);
		}
		
		return root;
	}
	
	public static BTNode flattenBST2SortedDLLInplace(BTNode root){
		if(root == null){
			return null;
		}
		
		//convert left subtree to DLL and connect last node (=predecessor of current root) to current root 
		if(root.left != null){
			//convert left subtree
			BTNode left = flattenBST2SortedDLLInplace(root.left);
			
			//find last node of the left DLL
			while(left.right != null){
				left = left.right;
			}
			
			//connect left DLL to root
			left.right = root;
			root.left = left;
		}
		//convert right subtree to DLL and connect root to the first node (=successor of current root)
		if(root.right != null){
			//convert left subtree
			BTNode right = flattenBST2SortedDLLInplace(root.right);
			
			//find first node of the left DLL
			while(right.left != null){
				right = right.left;
			}
			
			//connect left DLL to root
			right.left = root;
			root.right = right;
		}
		
		return root;
	}
	
	public static BTNode flattenBST2SortedCircularDLLInplace(BTNode root){
		if(root == null){
			return null;
		}
		
		//recursively divide it into left and right subtree until we get a leaf node which 
		//can be a stand alone doubly circular linked list by doing some simple pointer manipulation
		BTNode left = flattenBST2SortedCircularDLLInplace(root.left);
		BTNode right = flattenBST2SortedCircularDLLInplace(root.right);
		
		//Let's first convert the root node into a stand alone circular DLL - just make it a self loop
		root.right = root;
		root.left = root;
		
		//We have now sublist on the left of root and sublist on the right of root.
		//So, we just need to append left, root, and right sublists in the respective order
		left = concatCircularDLL(left, root);
		left = concatCircularDLL(left, right);
		
		return left;
	}
	
	public static void joinCircularNodes(BTNode node1, BTNode node2){
		node1.right = node2;
		node2.left = node1;
	}
	
	//concats head2 list to the end of head1 list
	public static BTNode concatCircularDLL(BTNode head1, BTNode head2){
		if(head1 == null){
			return head2;
		}
		if(head2 == null){
			return head1;
		}
		//in order to concat two circular list we need to
		//1. join tail1 and head2 to append list2 to at the end of list1
		//2. join tail2 and head1 to make it circular
		BTNode tail1 = head1.left;
		BTNode tail2 = head2.left;
		
		//join tail1 and head2 to append list2 to at the end of list1
		joinCircularNodes(tail1, head2);
		//join tail2 and head1 to make it circular
		joinCircularNodes(tail2, head1);
		
		return head1;
	}
	
	public static void InorderTraversal(BTNode root){
		Stack<BTNode> stack = new Stack<test.BTNode>();
		
		BTNode cur = root;
		while(true){
			//go to left most node while pushing nodes along the way
			if(cur != null){
				stack.push(cur);
				cur = cur.left;
			}
			else{
				//backtrack
				if(!stack.isEmpty()){
					cur = stack.pop();
					System.out.print(cur.key+" ");
					cur = cur.right;
				}
				else{
					return;//done
				}
			}
		}
	}
	
	public static void MorrisInorderTraversal(BTNode root){
		if(root == null){
			return;
		}
		
		BTNode cur = root;
		BTNode pre = null;
		while(cur != null){
			//if no left subtree the visit right subtree right away after printing current node
			if(cur.left == null){
				System.out.print(cur.key+", ");
				cur = cur.right;
			}
			else{
				//otherwise we will traverse the left subtree and come back to current 
				//node by using threaded pointer from predecessor of current node 
				//first find the predecessor of cur
				pre = cur.left;
				while(pre.right != null && pre.right != cur){
					pre = pre.right;
				}
				
				//threaded pointer not added - add it and go to left subtree to traverse
				if(pre.right == null){
					pre.right = cur;
					cur = cur.left;
				}
				else{
					//we traversed left subtree through threaded pointer and reached cur again
					//so revert the threaded pointer and print out current node before traversing right subtree
					pre.right = null;
					System.out.print(cur.key+", ");
					//now traverse right subtree
					cur = cur.right;
				}
			}
		}
	}
	
	public static ListNode convertToList(BTNode root){
		if(root == null){
			return null;
		}
		
		ListNode head = null;;
		ListNode iterator = head;
		BTNode cur = root;
		BTNode pre = null;
		while(cur != null){
			//if no left subtree the visit right subtree right away after printing current node
			if(cur.left == null){
				if(head == null){
					head = new ListNode(cur.key);
					iterator = head;
				}
				else{
					iterator.next = new ListNode(cur.key);
					iterator = iterator.next;
				}
				System.out.print(cur.key+", ");
				cur = cur.right;
			}
			else{
				//otherwise we will traverse the left subtree and come back to current 
				//node by using threaded pointer from predecessor of current node 
				//first find the predecessor of cur
				pre = cur.left;
				while(pre.right != null && pre.right != cur){
					pre = pre.right;
				}
				
				//threaded pointer not added - add it and go to left subtree to traverse
				if(pre.right == null){
					pre.right = cur;
					cur = cur.left;
				}
				else{
					//we traversed left subtree through threaded pointer and reached cur again
					//so revert the threaded pointer and print out current node before traversing right subtree
					pre.right = null;
					if(head == null){
						head = new ListNode(cur.key);
						iterator = head;
					}
					else{
						iterator.next = new ListNode(cur.key);
						iterator = iterator.next;
					}
					System.out.print(cur.key+", ");
					//now traverse right subtree
					cur = cur.right;
				}
			}
		}
		
		return head;
	}
	
	public static BTNode convertToListInplace(BTNode root){
		if(root == null){
			return null;
		}
		BTNode iterator = null;
		BTNode head = null;
		BTNode cur = root;
		BTNode pre = null;
		while(cur != null){
			//if no left subtree the visit right subtree right away after printing current node
			if(cur.left == null){
				if(head == null){
					head = cur;
					iterator = head;
				}
				else{
					iterator.right = cur;
					iterator = iterator.right;
				}
				System.out.print(cur.key+", ");
				cur = cur.right;
			}
			else{
				//otherwise we will traverse the left subtree and come back to current 
				//node by using threaded pointer from predecessor of current node 
				//first find the predecessor of cur
				pre = cur.left;
				while(pre.right != null && pre.right != cur){
					pre = pre.right;
				}
				
				//threaded pointer not added - add it and go to left subtree to traverse
				if(pre.right == null){
					pre.right = cur;
					cur = cur.left;
				}
				else{
					//we traversed left subtree through threaded pointer and reached cur again
					//so revert the threaded pointer and print out current node before traversing right subtree
					pre.right = null;
					if(head == null){
						head = cur;
						iterator = head;
					}
					else{
						iterator.right = cur;
						iterator = iterator.right;
					}
					System.out.print(cur.key+", ");
					//now traverse right subtree
					cur = cur.right;
				}
			}
		}
		
		return head;
	}
	
	public static boolean matchesFirst(String str, String pat){
		if(str == null){
			return pat == null;
		}
		if(pat == null){
			return str == null;
		}
		return (str.length() > 0 && str.charAt(0) == pat.charAt(0)) || (pat.charAt(0) == '.' && !str.isEmpty());  
	}
	
	public static boolean isMatch(String str, String pat){
		//base cases
		if(str == null){
			return pat == null;
		}
		else if(pat == null){
			return str == null;
		}
		if(str.isEmpty()){
			return pat.isEmpty();
		}
		
		//pattern without *
		if((pat.length() == 1 && pat.charAt(0) != '*' ) || pat.length() > 1 && pat.charAt(1) != '*'){
			//must match the first character
			if(!matchesFirst(str, pat)){
				return false;
			}
			//match rest
			String restStr = str.length() > 1 ? str.substring(1) : null;
			String restPat = pat.length() > 1 ? pat.substring(1) : null;
			return isMatch(restStr, restPat);
		}
		//pattern with * (0 or more matches)
		else{
			//zero match of first character of the pattern
			String rigtpat = pat.length() > 2 ? pat.substring(2) : null;
			if(isMatch(str, rigtpat)){
				return true;
			}
			//Otherwise match all possible length prefix of str to match and return true if any match found
			while(matchesFirst(str, pat)){
				str= str.length() > 1 ? str.substring(1) : null;
				if(isMatch(str, rigtpat)){
					return true;
				}
			}
		}
		
		return false;
	}
	
	public static boolean wordBreak(Set<String> dictionary, String text){
		//base case
		if(text.isEmpty()){
			return true;
		}
		//break the string at i+1 such that prefix text[...i] is in dict and suffix text[i+1...] is breakable
		for(int i = 0; i<text.length(); i++){
			if(dictionary.contains(text.substring(0, i+1)) && wordBreak(dictionary, text.substring(i+1))){
				return true;
			}
		}
		
		return false;
	}
	
	public static boolean wordBreak(Set<String> dictionary, String text, ArrayList<String> result){
		//base case
		if(text.isEmpty()){
			return true;
		}
		//break the string at i+1 such that prefix text[...i] is in dict and suffix text[i+1...] is breakable
		for(int i = 0; i<text.length(); i++){
			if(dictionary.contains(text.substring(0, i+1)) && wordBreak(dictionary, text.substring(i+1), result)){
				result.add(0, text.substring(0, i+1));
				return true;
			}
		}
		
		return false;
	}
	
	public static HashMap<String, ArrayList<String>> wordBreakMap = new HashMap<String, ArrayList<String>>();
	public static ArrayList<String> wordBreakAll(Set<String> dictionary, String text){
		//if already computed the current substring text then return from map
		if(wordBreakMap.containsKey(text)){
			return wordBreakMap.get(text);
		}
		ArrayList<String>  result = new ArrayList<String>();
		
		//if the whole word is in the dictionary then we add this to final result
		if(dictionary.contains(text)){
			result.add(text);
		}
		
		//try each prefix and extend
		for(int i = 0; i< text.length(); i++){
			String prefix = text.substring(0, i+1);
			if(dictionary.contains(prefix)){
				//extend
				String suffix = text.substring(i+1);
				ArrayList<String> subRes = wordBreakAll(dictionary, suffix);
				for(String word : subRes){
					result.add(prefix+" "+word);
				}
			}
		}
		
		wordBreakMap.put(text, result);
		return result;
	}
	
	public static boolean wordBreakDP(Set<String> dictionary, String text){
		int n = text.length();
		if(n == 0){
			return true;
		}
		
		//dp[i] = true if there is a solution in prefix text[0..i]
		boolean[] dp = new boolean[n];  
		
		//try all possible prefixes
		for(int i = 0; i< n; i++){
			//check from dp if current length prefix is a solution
			//if not then the prefix should be present in dictionary
			if(dp[i] == false && dictionary.contains(text.substring(0, i+1))){
				dp[i] = true;
			}
			
			//if this prefix contains in dictionary the try to extend the prefix up to end of the string
			if(dp[i] == true){
				for(int j = i+1; j < n; j++){
					//check id dp[j] already computed to a solution , 
					//other wise we need to check if text[i+1..i] contains in the dict.
					//so that we can create a longer prefix text[0..j]
					if(dp[j] == false){
						dp[j] = dictionary.contains(text.substring(i+1, j+1));
					}
				}
			}
		}
		
		return dp[n-1];
	}
	
	public static void topologicalSort(int u, ArrayList<ArrayList<Integer>> adjList, int[] visited, Stack<Integer> stack){
		//mark as visited
		visited[u] = 1;
		
		//first visit all the neighbors to ensure topo sort order
		for(int v : adjList.get(u)){
			if(visited[v] == 0){
				topologicalSort(v, adjList, visited, stack);
			}
		}
		
		stack.add(u);
	}
	
	public static void topologicalSort(ArrayList<ArrayList<Integer>> adjList){
		int[] visited = new int[adjList.size()];
		Stack<Integer> stack = new Stack<Integer>();
		
		for(int i = 0; i<adjList.size(); i++){
			if(visited[i] == 0){
				topologicalSort(i, adjList, visited, stack);
			}
		}
		
		System.out.print("topo sort: ");
		while(!stack.isEmpty()){
			System.out.print(stack.pop()+" ");
		}
		System.out.println()	;
	}
	
	public static BTNode closestLeaf(BTNode root, int[] len, BTNode closest){
		if(root == null){
			return closest;
		}
		
		len[0]++;
		closest = closestLeaf(root.left, len, closest);
		if(root.left == null && root.right == null){
			if(len[0] < len[1]){
				len[1] = len[0];
				closest = root;
			}
		}
		len[0]--;
		closest = closestLeaf(root.right, len, closest);
		
		return closest;
	}
	
	//smallest lexicographic string after removing duplicates 
	public static String lexicoSmallNoDuplicates(String str){
		int[] hist = new int[256];
		StringBuilder out = new StringBuilder();
		
		//compute character count histogram
		for(int i = 0; i<str.length(); i++){
			hist[str.charAt(i)-'0']++;
		}
		
		//scan left to right and remove current if and only if - 
		//count for cur character is > 1 and value of character is lexicographically 
		//greater than next character. Otherwise we take the character (if not already taken early)
		for(int i = 0; i<str.length()-2; i++){
			int cur = str.charAt(i)-'0';
			int next = str.charAt(i+1)-'0';
			if(cur > next && hist[cur] > 1){
				hist[cur]--;
			}
			else if(hist[cur] != 0){
				out.append(str.charAt(i));
				hist[cur] = 0;
			}
		}
		
		if(hist[str.charAt(str.length()-1)-'0'] != 0){
			out.append(str.charAt(str.length()-1));
		}
		
		return out.toString();
	}
	
	public static int maxStockProfit(int[] price)
	{
		int maxProfit = 0;
		int minBuy = price[0];
		int tempStart = 0;
		int start = 0;
		int end = 0;
		
		for(int i=0; i<price.length;i++)
		{
			if(price[i] < minBuy)
			{
				minBuy = price[i];
				tempStart = i;
			}
			if((price[i] - minBuy) > maxProfit)
			{
				maxProfit = price[i] - minBuy;
				start = tempStart;
				end = i;
			}
		}
		
		return maxProfit;
	}
	
	interface JSONTokenStream {
		  boolean hasNext();
		  JSONToken next();
	}
	
	interface JSONToken {
	  int type(); // 0=StartObject, 1=EndObject, 2=Field
	  String name();        // null for EndObject
	  String value();       // null for StartObject, EndObject
	}
	
	boolean equals(JSONTokenStream s1, JSONTokenStream s2) {
	    
	    Node node1 = null;
	    Node node2 = null;
	    Stack<Node> lastRoot = new Stack<test.Node>();
	    while(s1.hasNext()){
	    	JSONToken cur = s1.next();
	        if(node1 == null && cur.type() == 0){
	            node1 = new Node(cur);
	            lastRoot.push(node1);
	        }
	        else if(node1 != null && cur.type() == 1){
	        	lastRoot.pop();
	        }
	        else{
	            lastRoot.peek().addChild(cur);
	        }
	    }
	    
	    lastRoot = new Stack<test.Node>();
	    while(s2.hasNext()){
	    	JSONToken cur = s2.next();
	        if(node2 == null && cur.type() == 0){
	            node2 = new Node(cur);
	            lastRoot.push(node2);
	        }
	        else if(node2 != null && cur.type() == 1){
	        	lastRoot.pop();
	        }
	        else{
	            lastRoot.peek().addChild(cur);
	        }
	    }
	    
	    return node1.equals(node2);
	    
	}
	
	class Node{
	    
	    int type;
	    String name;
	    String value;
	    TreeSet<Node> child;
	    
	    public Node(JSONToken token){
	        name = token.name();
	        value = token.value();
	        type = token.type();
	        child = new TreeSet<Node>();
	    }
	    
	    public void addChild(JSONToken tok){
	        Node child = new Node(tok);
	        this.child.add(child);
	    }
	    
	    public boolean equals(Node n1, Node n2){
	    	return (n1.type == n2.type) 
	    			&& (n1.name.equals(n2.name)) 
	    			&& (n1.value == n2.value) 
	    			&& (n1.child.size() == n2.child.size()) 
	    			&& (n1.child.equals(n2.child));
	    }
	    
	}
	
	public static int longestValidParenthesis(String str){
		int maxLen = 0;
		int start = -1;
		Stack<Integer> stack = new Stack<Integer>();
		
		for(int i = 0; i<str.length(); i++){
			if(str.charAt(i) == '('){
				stack.push(i);
			}
			else{
				//no matching left
				if(stack.isEmpty()){
					start = i;
				}
				else{
					stack.pop();
					if(!stack.isEmpty()){
						maxLen = Math.max(maxLen, i-stack.peek());
					}
					else{
						maxLen = Math.max(maxLen, i-start);
					}
				}
			}
		}
		
		return maxLen;
	}
	
	public static int longestValidParenthesis(String str, int dir){
		int start = 0;
		int end = str.length()-1;
		int openChar = '(';
		int count = 0;
		int maxLen = 0;
		int curLen = 0;
		if(dir == -1){
			start = end;
			end = 0;
			openChar = ')';
		}
		
		for(int i = start; i!=end; i+=dir){
			if(str.charAt(i) == openChar){
				count++;
			}
			else{
				//no matching left
				if(count <= 0){
					//restart
					curLen = 0;
				}
				else{
					//a local match
					count--;
					curLen += 2;
					
					if(count == 0){
						maxLen = Math.max(maxLen, curLen);
					}
				}
			}
		}
		
		return maxLen;
	}
	
	public static int longestValidParenthesis2(String str){
		return Math.max(longestValidParenthesis(str, 1), longestValidParenthesis(str, -1));
	}
	
	public static int perfectSquareDP(int n){
		if(n <= 0){
			return 0;
		}
		
		int[] dp = new int[n+1];
		Arrays.fill(dp, Integer.MAX_VALUE);
		dp[0] = 0;
		dp[1] = 1;
		
		//to compute least perfect for n we compute top down for each 
		//possible value sum from 2 to n
		for(int i = 2; i<=n; i++){
			//for a particular value i we can break it as sum of a perfect square j*j and 
			//all perfect squares from solution of the remainder (i-j*j)
			for(int j = 1; j*j<=i; j++){
				dp[i] = Math.min(dp[i], 1+dp[i-j*j]);
			}
		}
		
		return dp[n];
	}
	
	private static boolean is_square(int n){  
	    int sqrt_n = (int)(Math.sqrt(n));  
	    return (sqrt_n*sqrt_n == n);  
	}
		
	// Based on Lagrange's Four Square theorem, there 
	// are only 4 possible results: 1, 2, 3, 4.
	public static int perfectSquaresLagrange(int n) 
	{  
	    // If n is a perfect square, return 1.
	    if(is_square(n)) 
	    {
	        return 1;  
	    }
	
	    // The result is 4 if n can be written in the 
	    // form of 4^k*(8*m + 7).
	    while ((n & 3) == 0) // n%4 == 0  
	    {
	        n >>= 2;  
	    }
	    if ((n & 7) == 7) // n%8 == 7
	    {
	        return 4;
	    }
	
	    // Check whether 2 is the result.
	    int sqrt_n = (int)(Math.sqrt(n)); 
	    for(int i = 1; i <= sqrt_n; i++)
	    {  
	        if (is_square(n - i*i)) 
	        {
	            return 2;  
	        }
	    }  
	
	    return 3;  
	}  
	
	public static BTNode inorderInPlaceUsingStack(BTNode root){
		
		Stack<BTNode> stack = new Stack<test.BTNode>();
		BTNode cur = root;
		BTNode head =  null;
		BTNode iterator = null;
		
		while(true){
			if(cur != null){
				stack.push(cur);
				cur = cur.left;
			}
			else{
				if(stack.isEmpty()){
					return head;
				}
				else{
					cur = stack.pop();
					if(head == null){
						head = cur;
						iterator = head;
					}
					else if(iterator != null){
						iterator.right = cur;
						iterator = iterator.right; 
					}
					cur = cur.right;
				}
			}
		}
	}
	
	public static void rotateRight(int[] A, int k){
		int n = A.length;
		if(n <= 1){
			return;
		}
		
		k = k%n;
		
		if(k == 0){
			return;
		}
		
		//reverse non rotated part
		reverse(A, 0, n-k-1);
		//reverse rotated part
		reverse(A, n-k, n-1);
		//reverse the whole array
		reverse(A, 0, n-1);
	}
	
	public static void rotateLeft(int[] A, int k){
		int n = A.length;
		if(n <= 1){
			return;
		}
		
		k = k%n;
		
		if(k == 0){
			return;
		}
		
		//reverse the whole array
		reverse(A, 0, n-1);
		//reverse rotated part
		reverse(A, n-k, n-1);
		//reverse non rotated part
		reverse(A, 0, n-k-1);
	}
	
//	public static void rotateRight(int[] A, int d){
//		int n = A.length;
//		d = d%n;
//		if(n <= 1 || d == 0){
//			return;
//		}
//		
//		int i = 0;
//		int k = n - d;
//		int j = k;
//		
//		if(k < n/2){
//			rotateLeft(A, n-d);
//			return;
//		}
//		
//		while(i < j){
//			swap(A, i, j);
//			i++;
//			j = (j == n-1) ? k : j+1;
//		}
//	}
//	
//	public static void rotateLeft(int[] A, int d){
//		int n = A.length;
//		d = d%n;
//		
//		if(n <= 1 || d == 0){
//			return;
//		}
//		
//		int i = n-1;
//		int k = d-1;
//		int j = k;
//		
//		if(k > n/2){
//			rotateRight(A, d);
//			return;
//		}
//		
//		while(j < i){
//			swap(A, i, j);
//			i--;
//			j = (j == 0) ? k : j-1;
//		}
//	}
	
	public static int maxRectangleAreaHistogram(int hist[]){
		int maxArea = 0;
		Stack<Integer> stack = new Stack<Integer>();
		int n = hist.length;
		int i = 0;
		
		while(i < n){
			//push a new bar if - 
			//1. it is in current subhistogram if monotonic decreasing bars, or
			//2. we found a start of a new subhistogram
			if(stack.isEmpty() || stack.peek() <= hist[i]){
				stack.push(i++);
			}
			//we found a bar in monotonic decreasing order while we already have a start bar on the stack
			//so, compute area upto to the last pushed bar
			
			else{
				//pop the last pushed bar
				int lastBarIndex = stack.pop();
				int height = hist[lastBarIndex];
				
				//leftmost bar in which can be either - 
				//1. first bar if no bar on the stack, or
				//2. the top bar on the stack 
				int leftMaxBarIndex = stack.isEmpty() ? 0 : stack.peek();
				
				//right index is always current index
				int rightBarIndex = i;
				
				//width between right bar and left bar
				int width = rightBarIndex - leftMaxBarIndex - 1;
				
				//compute area and update max area
				maxArea = Math.max(maxArea, height*width);
			}
		}
		
		while(!stack.isEmpty()){
			//pop the last pushed bar
			int lastBarIndex = stack.pop();
			int height = hist[lastBarIndex];
			
			//leftmost bar in which can be either - 
			//1. first bar if no bar on the stack, or
			//2. the top bar on the stack 
			int leftMaxBarIndex = stack.isEmpty() ? 0 : stack.peek();
			
			//right index is always current index
			int rightBarIndex = i;
			
			//width between right bar and left bar
			int width = rightBarIndex - leftMaxBarIndex - 1;
			
			//compute area and update max area
			maxArea = Math.max(maxArea, height*width);
		}
		
		return maxArea;
	}
	
	public static int minDiffElements(int a1[], int a2[]){
		int minDiff = Integer.MAX_VALUE;
		int min1 = -1;
		int min2 = -1;
		int i = 0;
		int j = 0;
		int n1 = a1.length;
		int n2 = a2.length;
		int diff = 0;
		
		Arrays.sort(a1);
		Arrays.sort(a2);
		while(i < n1 && j < n2){
			diff = Math.abs(a1[i]-a2[j]);
			if(diff < minDiff){
				minDiff = diff;
				min1 = a1[i];
				min2 = a2[j];
			}
			
			if(a1[i] < a2[j]){
				i++;
			}
			else{
				j++;
			}
		}
		
		System.out.println("min diff between two array elements: between "+min1+" and "+min2+" min diff: "+minDiff);
		return minDiff;
	}
	
	public static int maxSumSubSeqNonContagious(int a[]){
		int max_include[] = new int[a.length];
		int max_exclude[] = new int[a.length];
		max_include[0] = a[0];
		max_exclude[0] = Integer.MIN_VALUE;
		int max = a[0];
		
		for(int i = 1; i<a.length; i++){
			max_include[i] = Math.max(max_exclude[i-1]+a[i], a[i]);
			max_exclude[i] = Math.max(max_include[i-1], max_exclude[i-1]);
			max = Math.max(max_include[i], max_exclude[i]);
		}
		
		return max;
	}
	
	public static class Point implements Comparable<Point> {
	    public double x;
	    public double y;
	
	    public Point(final double x, final double y) {
	        this.x = x;
	        this.y = y;
	    }
	    
	    public double getDist(){
	    	return x*x+y*y;
	    }
	
		@Override
		public int compareTo(Point o) {
			int c = Double.compare(getDist(), o.getDist());
			if(c == 0){
				c = Double.compare(x, o.x);
				if(c == 0){
					c = Double.compare(y, o.y);
				}
			}
			
			return c;
		}
	
		@Override
		public String toString() {
			return "(" + x + "," + y + ")";
		}
	}
	
	public static Point[] closestk(final Point points[], final int k) {
	    //max heap
	    final PriorityQueue<Point> kClosest = new PriorityQueue<>(k, Collections.reverseOrder());
	
	    for (int i = 0; i < points.length; i++) {
	        if (kClosest.size() < k) {
	        	kClosest.add(points[i]);
	        } else if (points[i].getDist() < kClosest.peek().getDist()) {
	            kClosest.remove();
	            kClosest.add(points[i]);
	        }
	    }
	
	    return kClosest.toArray(new Point[k]);
	}
	
	public static int searchInSortedRotatedArray(int a[], int key){
		int l = 0;
		int h = a.length-1;
		int mid;
		
		while(l < h){
			mid = (l+h)/2;
			if(a[mid] == key){
				return mid;
			}
			//search in left subtree
			else if(a[mid] > key){
				//if left subtree has rotated part
				if(a[l] > key){
					l = mid+1; 
				}
				//otherwise its in sorted part
				else{
					h = mid-1;
				}
			}
			else{
				//if right subtree has rotated part
				if(a[h] < key){
					h = mid-1;
				}
				//otherwise its in sorted part
				else{
					l = mid+1;
				}
			}
		}
		
		return -1;
	}
	
	public static int searchRotationPosition(int a[]){
		int n = a.length;
		int l = 0;
		int h = a.length-1;
		int mid;
		
		while(l < h){
			mid = (l+h)/2;
			
			if(mid > 0 && mid < n-1 && a[mid] < a[mid-1] && a[mid] <= a[mid+1]){
				return mid;
			}
			else if(a[mid] >= a[h]){
				l = mid+1;
			}
			else{
				h = mid - 1;
			}
		}
		
		return -1;
	}
	
	public static int findRotationPositin(final int[] a) {
	    if (a.length <= 1) {
	        return 0;
	    }

	    int l = 0;
	    int r = a.length - 1;
	    int m = (l + r) / 2;

	    while (a[l] > a[r]) {
	        m = (l + r) / 2;

	        if (a[m] > a[r]) {
	            l = m + 1;
	        } else {
	            r = m;
	        }
	    }

	    return l;
	}
	
	public static class MovingAvgLastN{
		int maxTotal;
		int total;
		double lastN[];
		double avg;
		int head;
		
		public MovingAvgLastN(int N){
			maxTotal = N;
			lastN = new double[N];
			avg = 0;
			head = 0;
			total = 0;
		}
		
		public void add(double num){
			double prevSum = total*avg;
			
			if(total == maxTotal){
				prevSum-=lastN[head];
				total--;
			}
			
			head = (head+1)%maxTotal;
			int emptyPos = (maxTotal+head-1)%maxTotal;
			lastN[emptyPos] = num;
			
			double newSum = prevSum+num;
			total++;
			avg = newSum/total;
		}
		
		public double getAvg(){
			return avg;
		}
	}
	
	public static void wiggleSort(int a[]){
		for(int i = 0; i<a.length; i++){
			int odd = i&1;
			if(odd == 1){
				if(a[i-1]>a[i]){
					swap(a, i-1, i);
				}
			}
			else{
				if(i!=0 && a[i-1]<a[i]){
					swap(a, i-1, i);
				}
			}
		}
	}
	
	public static String rearrangeAdjacentDuplicates(String str){
		final class CharFreq implements Comparable<CharFreq>{
			char c;
			int freq;
			public CharFreq(char ch, int count){
				c = ch;
				freq = count;
			}
			@Override
			public int compareTo(CharFreq o) {
				int comp = Double.compare(freq, o.freq);
				if(comp == 0){
					comp = Character.compare(o.c, c);
				}
				
				return comp;
			}
		}
		
		int n = str.length();
		StringBuffer rearranged = new StringBuffer();
		PriorityQueue<CharFreq> maxHeap = new PriorityQueue<CharFreq>(256, Collections.reverseOrder());
		int freqHistoGram[] = new int[256];
		//build the character frequency histogram
		for(char c : str.toCharArray()){
			freqHistoGram[c]++;
			
			//if a character repeats more than n/2 then we can't rearrange
			if(freqHistoGram[c] > (n+1)/2){
				return str;
			}
		}
		//build the max heap of histogram
		for(char i  = 0; i < 256; i++){
			if(freqHistoGram[i] > 0)
				maxHeap.add(new CharFreq(i, freqHistoGram[i]));
		}
		
		//rearrange - pop top 2 most frequent items and arrange them in adjacent positions
		//decrease the histogram frequency of the selected chars 
		while(!maxHeap.isEmpty()){
			//extract top one and decrease the hstogram by one
			CharFreq first = maxHeap.poll();
			rearranged.append(first.c);
			first.freq--;
			
			CharFreq second = null;
			//extract second top and decrease the histogram by one
			if(!maxHeap.isEmpty()){
				second = maxHeap.poll();
				rearranged.append(second.c);
				second.freq--;
			}
			
			//add back the updated histograms 
			if(first.freq > 0){
				maxHeap.add(first);
			}
			if(second != null && second.freq > 0){
				maxHeap.add(second);
			}
		}
		
		return rearranged.toString();
	}
	
	public static class Codec {
	
	    // Encodes a tree to a single string.
	    public String serialize(TreeNode root) {
	        if(root == null){
	            return "null";
	        }
	        
	        StringBuffer serTree = new StringBuffer();
	        Queue<TreeNode> queue = new LinkedList<TreeNode>();
	        queue.add(root);
	        TreeNode node;
	        
	        while(!queue.isEmpty()){
	            node = queue.poll();
	            if(node.key == Integer.MIN_VALUE){
	                serTree.append("null,");
	                continue;
	            }
	            else{
	                serTree.append(node.key+",");
	            }
	            
	            if(node.left != null){
	                queue.add(node.left);
	            }
	            else{
	                queue.add(new TreeNode(Integer.MIN_VALUE));
	            }
	            
	            if(node.right != null){
	                queue.add(node.right);
	            }
	            else{
	                queue.add(new TreeNode(Integer.MIN_VALUE));
	            }
	        }
	        
	        return serTree.toString();
	    }
	
	    // Decodes your encoded data to tree.
	    public TreeNode deserialize(String data) {
	        if(data == null || data.isEmpty() || data.startsWith("null")){
	            return null;
	        }
	        
	        String[] nodes = data.split(",");
	        if(nodes.length == 0){
	            return null;
	        }   
	        
	        TreeNode root = new TreeNode(0);
	        Queue<TreeNode> queue = new LinkedList<TreeNode>();
	        queue.add(root);
	        
	        TreeNode node;
	        int i = 0;
	        while(!queue.isEmpty()){
	            node = queue.poll();
	            node.key = Integer.parseInt(nodes[node.key]);
	
	            int left = i+1;
	            int right = i+2;
	            
	            if(left < nodes.length-1 && !nodes[left].equals("null")){
	                TreeNode leftNode = new TreeNode(left);
	                node.left = leftNode;
	                queue.add(leftNode);
	            }
	            if(right < nodes.length-1 && !nodes[right].equals("null")){
	                TreeNode rightNode = new TreeNode(right);
	                node.right = rightNode;
	                queue.add(rightNode);
	            }
	            
	            i+=2;
	        }
	        
	        return root;
	    }
	}
	
	public boolean searchMatrix2(int[][] matrix, int target) {
	    if(matrix == null && matrix.length == 0){
	        return false;
	    }
	    int n = matrix.length;
	    int m = matrix[0].length;
	    int r = m-1;
	    int t = 0;
	    
	    
	    while(t < n && r >=0){
	         if(matrix[t][r] == target){
	             return true;
	         }
	         else if(matrix[t][r] > target){
	             r--;
	         }
	         else{
	             t++;
	         }
	    }
	    
	    return false;
	}
	
	public boolean searchMatrix1(int[][] matrix, int target) {
	    if(matrix == null || matrix.length == 0){
	        return false;
	    }
	    
	    //find the row
	    int n = matrix.length;
	    int m = matrix[0].length;
	    
	    int row = -1;
	    int rl = 0;
	    int rh = n-1;
	    while(rl <= rh){
	        int mid = (rl+rh)/2;
	        if(matrix[mid][0] == target || matrix[mid][m-1] == target){
	            return true;
	        }
	        else if(matrix[mid][0] > target){
	            rh = mid-1;
	        }
	        else{
	            if(matrix[mid][m-1] > target){
	                row = mid;
	                break;
	            }
	            else{
	                rl = mid+1;
	            }
	        }
	    }
	    
	    if(row == -1){
	        return false;
	    }
	    
	    //search in the row
	    int col = -1;
	    int cl = 0;
	    int ch = m-1;
	    while(cl <= ch){
	        int mid = (cl+ch)/2;
	        if(matrix[row][mid] == target){
	            return true;
	        }
	        else if(matrix[row][mid] > target){
	            ch = mid-1;
	        }
	        else{
	            cl = mid+1;
	        }
	    }
	    
	    return false;
	}
	
	public static ArrayList<String> convertInfix(String[] exprTokens){
		ArrayList<String> infix = new ArrayList<String>();
		Stack<String> operatorStack = new Stack<String>();
		
		for(String op : exprTokens){
			op = op.trim();
			//if its an operand , simply append to output
			if(!isOperator(op)){
				infix.add(op);
			}
			//if its an operator
			else{
				//if its a left parenthesis then push it to stack
				if(op.equals("(")){
					operatorStack.push("(");
				}
				//other wise if it is a right parenthesis then pop the stack untill we see a matching left parenthesis
				else if(op.equals(")")){
					while(!operatorStack.peek().equals("(") && !operatorStack.isEmpty()){
						infix.add(operatorStack.pop());
					}
					
					//if we do not have a matching left parethesis then it's a malformed expression
					if(operatorStack.isEmpty() || !operatorStack.peek().equals("(")){
						return null;
					}
					//otherwise we found a matching left. Just pop it out
					else{
						operatorStack.pop();
					}
				}
				//otherwise its an operator
				else{
					//keep poping out element from stack and append in output as long as we see a higher precedence operator 
					//in the top of stack
					while(
							!operatorStack.isEmpty() 
							&& (
									(isLeftAssociative(op) && getPrecedence(op) <= getPrecedence(operatorStack.peek()))
									|| (!isLeftAssociative(op) && getPrecedence(op) < getPrecedence(operatorStack.peek()))
							   )
					    ){
						infix.add(operatorStack.pop());
					}
					//ow add the operator
					operatorStack.push(op);
				}
			}
		}
		
		//if there are left over element sin stack then append them in the output
		while(!operatorStack.isEmpty()){
			infix.add(operatorStack.pop());
		}
		
		return infix;
	}
	
	private static int getPrecedence(String op){
		if(op.equals("+") || op.equals("-")){
			return 2;
		}
		if(op.equals("*") || op.equals("/")){
			return 3;
		}
		if(op.equals("^")){
			return 4;
		}
		
		return 0;
	}
	
	private static boolean isLeftAssociative(String op){
		if(op.equals("+") || op.equals("-") || op.equals("*") || op.equals("/")){
			return true;
		}
		if(op.equals("^")){
			return false;
		}
		
		return false;
	}
	
	private static boolean isOperator(String op){
		return op.matches("[-+*/^()]");
	}
	
	public static double evaluateInfix(ArrayList<String> infix){
		Stack<String> opStack = new Stack<String>();
		
		for(String op : infix){
			if(isOperator(op)){
				//pop second operand frst (because it's in stack)
				if(opStack.isEmpty()){
					return Double.NaN;
				}
				String op2 = opStack.pop();
				
				//pop first operand second (because it's in stack)
				if(opStack.isEmpty()){
					return Double.NaN;
				}
				String op1 = opStack.pop();
				
				//evaluate the expression
				double eval = evaluate(op1, op2, op);
				
				if(eval == Double.NaN){
					return Double.NaN;
				}
				
				//push the evaluated value to stack
				opStack.push(eval+"");
			}
			else{
				opStack.push(op);
			}
		}
		
		if(opStack.size() != 1){
			return Double.NaN;
		}
		
		return Double.parseDouble(opStack.pop());
	}
	
	private static double evaluate(String op1, String op2, String op){
		if(op.equals("+")){
			return Double.parseDouble(op1)+Double.parseDouble(op2);
		}
		else if(op.equals("-")){
			return Double.parseDouble(op1)-Double.parseDouble(op2);
		}
		else if(op.equals("*")){
			return Double.parseDouble(op1)*Double.parseDouble(op2);
		}
		else if(op.equals("/")){
			double denm = Double.parseDouble(op2);
			if(denm == 0){
				return Double.NaN;
			}
			else {
				return Double.parseDouble(op1)/Double.parseDouble(op2);
			}
		}
		else return Double.NaN;
	}
	
	public static int maxLoad(Job[] jobs){
		int maxLoad = 0;
		int curLoad = 0;
		
		Job[] start = Arrays.copyOf(jobs, jobs.length);
		Job[] end = Arrays.copyOf(jobs, jobs.length);
		
		Arrays.sort(start, new Comparator<Job>() {
	
			@Override
			public int compare(Job o1, Job o2) {
				return Integer.compare(o1.start, o2.start);
			}
		});
		Arrays.sort(end, new Comparator<Job>() {
	
			@Override
			public int compare(Job o1, Job o2) {
				return Integer.compare(o1.finish , o2.finish);
			}
		});
		
		int i = 0, j = 0;
		while(i < start.length && j < end.length){
			if(start[i].start <= end[j].finish){
				curLoad += start[i].weight;
				maxLoad = Math.max(maxLoad, curLoad);
				i++;
			}
			else{
				curLoad -= end[j].weight;
				j++;
			}
		}
		
		return maxLoad;
	}
	
	public static int countTriangleTriplets(int[] segments){
		int count = 0;
		int n = segments.length;
		Arrays.sort(segments);
		
		for(int i = 0; i<n-2; i++){
			int k = i+2;
			for(int j = i+1; j < n; j++){
				while(k < n && segments[i]+segments[j] > segments[k]){
					k++;
				}
				count += k-j-1;
			}
		}
		return count;
	}
	
	public static int maxProductSubArr(int a[]){
		int localMax = 1;
		int localMin = 1;
		int globalMaxProd = 1;
		
		for(int i = 0; i < a.length; i++){
			if(a[i] == 0){
				localMax = 1;
				localMin = 1;
			}
			else if(a[i] > 0){
				localMax *= a[i];
				localMin = Math.min(localMin*a[i],1);
			}
			else{
				int temp = localMin;
				localMin = Math.min(localMax*a[i], 1);
				localMax = Math.max(temp*a[i], 1);
			}
			
			globalMaxProd = Math.max(globalMaxProd, localMax);
		}
		
		return globalMaxProd;
	}
	
	public static int influencer(int[][] following){
		int influencer = 0;
		
		//find the candidate influencer by testing each person i
		//a person i may be a candidate influencer if s/he follows nobody or som
		for(int i = 1; i < following.length; i++){
			if(following[i][influencer] == 0 || following[influencer][i] == 1){
				influencer = i;
			}
		}
		
		//verify that the candidate influencer is indeed an influencer
		for(int i = 0; i < following.length; i++){
			if(i == influencer){
				continue;
			}
			//to be influencer he/she shouldn't follow anybody and there should be nobody else who doesn't follw him/her
			if(following[i][influencer] == 0 || following[influencer][i] == 1){
				return -1;
			}
		}
		
		return influencer;
	}
	
	public static int nthUglyNumber(int n){
		int nthUgly = 1;
		PriorityQueue<Integer> minHeap = new PriorityQueue<Integer>();
		Set<Integer> uniques = new HashSet<Integer>();
		minHeap.offer(1);
		
		while(n > 0){
			nthUgly = minHeap.poll();
			int next = nthUgly*2;
			if(nthUgly <= Integer.MAX_VALUE/2 && !uniques.contains(next)){
				minHeap.offer(next);
				uniques.add(next);
			}
			next = nthUgly*3;
			if(nthUgly <= Integer.MAX_VALUE/3 && !uniques.contains(next)){
				minHeap.offer(next);
				uniques.add(next);
			}
			next = nthUgly*5;
			if(nthUgly <= Integer.MAX_VALUE/5 && !uniques.contains(next)){
				minHeap.offer(next);
				uniques.add(next);
			}
			n--;
		}
		
		return nthUgly;
	}
	
	public static int nthUglyDP(int n){
		int merged[] = new int[n];
		//1 is considered as ugly so, its the first ugly number
		merged[0] = 1;
		//pointer to the three sets of ugly numbers generated by multiplying respectively by 2, 3, and 5
		//p2 points to current ugly of the sequence : 1*2, 2*2, 3*2, 4*2, ...
		//p3 points to current ugly of the sequence : 1*3, 2*3, 3*3, 4*3, ...
		//p5 points to current ugly of the sequence : 1*5, 2*5, 3*5, 4*5, ...
		int p2 = 0, p3 = 0, p5 = 0;
		
		//merge the 3 sequences pointed by p2, p3, and p5 and always take the min as we do in merge sort
		for(int i = 1; i<n; i++){
			merged[i] = Math.min(Math.min(merged[p2]*2, merged[p3]*3), merged[p5]*5);
			
			//now increment the corrsponding pointer - same number can be generated in multiple sequences
			//for example, 10 can be genetaed by 2 as 5*2 or by 5 as 2*5. So, we increment all pointers 
			//that contains same value to avoid duplicates
			if(merged[i] == merged[p2]*2){
				p2++;
			}
			if(merged[i] == merged[p3]*3){
				p3++;
			}
			if(merged[i] == merged[p5]*5){
				p5++;
			}
		}
		
		return merged[n-1];
	}
	
	public static void printFactors(int number) {
		printFactors("", number, number);
	}

	public static void printFactors(String expression, int dividend, int previous) {
		if (expression == "")
			System.out.println(previous + " * 1");
		for (int factor = dividend - 1; factor >= 2; --factor) {
			if (dividend % factor == 0 && factor <= previous) {
				int next = dividend / factor;
				if (next <= factor)
					if (next <= previous)
						System.out.println(expression + factor + " * " + next);
				printFactors(expression + factor + " * ", next, factor);
			}
		}
	}
	
	public static class Building{
		int l;
		int h;
		int r;
		
		public Building(int left, int height, int right){
			l = left;
			h = height;
			r = right;
		}
	}
	
	public static class Strip{
		int l;
		int h;
		
		public Strip(int left, int height){
			l = left;
			h = height;
		}
	
		@Override
		public String toString() {
			return "(" + l + ", " + h + ")";
		}
	}
	
	public static ArrayList<Strip> skyLine(Building[] buildings){
		int n = buildings.length;
		Building[] start = Arrays.copyOf(buildings, n);
		Building[] end = Arrays.copyOf(buildings, n);
		PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(n, Collections.reverseOrder());
		ArrayList<Strip> strips = new ArrayList<Strip>();
		
		//sort based on left coordinate of a building i.e. start of a range
		Arrays.sort(start, new Comparator<Building>() {
	
			@Override
			public int compare(Building o1, Building o2) {
				int c = Integer.compare(o1.l, o2.l);
    			if(c == 0){
    			    c = Integer.compare(o2.h, o1.h);
    			}
    			return c;
			}
		});
		
		//sort based on right coordinate of a building i.e. end of a range
		Arrays.sort(end, new Comparator<Building>() {
	
			@Override
			public int compare(Building o1, Building o2) {
				return Integer.compare(o1.r, o2.r);
			}
		});
		
		int i = 0, j  = 0;
		while(i < n || j < n){			
			//a new overlapping range i.e. a building
			if(i < n && start[i].l <= end[j].r){
				//update max height seen so far in current overlap
				maxHeap.add(start[i].h);
				//max height in current overlap including the current building
				int maxHeightIncldingMe = maxHeap.isEmpty() ? 0 : maxHeap.peek();
				//add th current strip with the left of building and max height seen so far in currne overlap
				strips.add(new Strip(start[i].l, maxHeightIncldingMe));
				//try next building
				i++;
			}
			else{
				//it's an end of a range of current overlap. So, we need to remove the height
				//of this range i.e. building from the max heap
				maxHeap.remove(end[j].h);
				//max height of remaining buildings in current overlap
				int maxHeightExcldingMe = maxHeap.isEmpty() ? 0 : maxHeap.peek();
				//add the current strip with the right of building and max height of remaining buildings
				strips.add(new Strip(end[j].r, maxHeightExcldingMe));
				//update end index
				j++;
			}
		}
		
		//merge strips to remove successive strips with same height
		ArrayList<Strip> mergedStrips = new ArrayList<Strip>();
		int prevHeight = 0;
		for(Strip st : strips){
			if(st.l == end[n-1].r && st.h != 0){
    	        continue;
    	    }
			if(prevHeight == 0){
				prevHeight = st.h;
				mergedStrips.add(st);
			}
			else if(prevHeight != st.h){
				prevHeight = st.h;
				mergedStrips.add(st);
			}
		}
		
		return mergedStrips;
	}
	
	public static boolean isBSTPostOrder(int[] a, int p, int q){
		int n = q-p+1;;
		//base case always true for 1 element
		if(n < 2){
			return true;
		}
		
		//partition into left subtree a[p..right-1] and right subtree a[right..q-1]
		int right = p;
		while(a[right] < a[q]) right++;
		
		//check validity of right subtree
		int i = right;
		while(i < q && a[i] > a[q]) i++;
		if(i < q){
			return false;
		}
		
		return isBSTPostOrder(a, p, right-1) && isBSTPostOrder(a, right, q-1);
	}
	
	public static int maxSetBitsSingleSegmentFlipped(int n){
		ArrayList<Integer> bits = new ArrayList<Integer>();
		while(n > 0){
			bits.add(0, n%2);
			n = n/2;
		}
		
		//kadanes algorithm
		int localMinima = 0;
		int globalMinima = 0;
		int zeroCount = 0;
		int localOnesToFlip = 0;
		int globalOnesToFlip = 0;
		int localStart = 0;
		int globalStart = 0;
		int end = 0;
		int totalOnes = 0;
		
		for(int i = 0; i< bits.size(); i++){
			if(bits.get(i) == 0){
				localMinima += -1;
				zeroCount++;
			}
			else{
				localMinima += 1;
				localOnesToFlip++;
				totalOnes++;
			}
			
			if(localMinima < globalMinima){
				globalMinima = localMinima;
				globalStart = localStart;
				end = i;
				globalOnesToFlip = localOnesToFlip;
			}
			else if(localMinima > 0){
				localMinima = 0;
				localStart = i+1;
				localOnesToFlip = 0;
			}
		}
		
		return zeroCount > 0 ? (end - globalStart +1) - globalOnesToFlip + totalOnes - globalOnesToFlip : totalOnes;
	}
	
	private static int binaryToDecimal(String str){
		int dec = 0;
		
		int b = 1; // 2^0
		for(int i = str.length()-1; i >= 0 ; i--){
			dec += b*(str.charAt(i)-'0');
			b*=2;
		}
		
		return dec;
	}
	
	public static void permuteList(String[][] list, int start, ArrayList<String> perms){
		if(start == list.length){
			if(perms.size() == list.length)
				System.out.println(perms.toString());
			return;
		}
		
		for(int i = 0; i < list[start].length; i++){
			perms.add(list[start][i]);
			for(int j = start+1; j <= list.length; j++){
				permuteList(list, j, perms);
			}
			perms.remove(list[start][i]);
		}
	}
	
	public static boolean checkDuplicateWithinK(int[] a, int k){
		int n = a.length;
		k = Math.min(n, k);
		
		Set<Integer> slidingWindow = new HashSet<Integer>(k);
		
		//create initial wiindow of size k
		int i;
		for(i = 0; i < k; i++){
			if(slidingWindow.contains(a[i])){
				return true;
			}
			
			slidingWindow.add(a[i]);
		}
		
		//now slide
		for(i = k; i < n; i++){
			slidingWindow.remove(a[i-k]);
			if(slidingWindow.contains(a[i])){
				return true;
			}
			slidingWindow.add(a[i]);
		}
		
		return false;
	}
	
	public static boolean checkDuplicateWithinK(int[][] mat, int k){
		class Cell{
			int row;
			int col;
			public Cell(int r, int c){
				this.row = r;
				this.col = c;
			}
		}
		
		int n = mat.length;
		int m = mat[0].length;
		k = Math.min(k, n*m);
		
		//map from distance to cell postions of the matrix
		Map<Integer, Set<Cell>> slidingWindow = new HashMap<Integer, Set<Cell>>();
		
		for(int i = 0; i < n; i++){
			for(int j = 0; j < m; j++){
				if(slidingWindow.containsKey(mat[i][j])){
					for(Cell c : slidingWindow.get(mat[i][j])){
						int manhattanDist = Math.abs(i - c.row)+Math.abs(j - c.col);
						
						if(manhattanDist <= k){
							return true;
						}
						
						if(i - c.row > k){
							slidingWindow.remove(c);
						}
					}
					
					slidingWindow.get(mat[i][j]).add(new Cell(i, j));
				}
				else{
					slidingWindow.put(mat[i][j], new HashSet<Cell>());
					slidingWindow.get(mat[i][j]).add(new Cell(i, j));
				}
			}
		}
		
		return false;
	}
	
	public static int maxGap(int[] a){
		int n = a.length;
		if(n < 2){
			return 0;
		}
		
		int max = Integer.MIN_VALUE;
		int min = Integer.MAX_VALUE;
		
		for(int i = 0; i < n; i++){
			max = Math.max(max, a[i]);
			min = Math.min(min, a[i]);
		}
		
		//n-1 buckets -  we only care about max and min in each buckets
		int[] bucketMaxima = new int[n-1];
		Arrays.fill(bucketMaxima, Integer.MIN_VALUE);
		int[] bucketMinima = new int[n-1];
		Arrays.fill(bucketMinima, Integer.MAX_VALUE);
		//bucket width
		float delta = (float)(max-min)/((float)n-1);
		
		//populate the bucket maxima and minima
		for(int i = 0; i < n; i++){
			if(a[i] == max || a[i] == min){
				continue;
			}
			
			int bucketIndex = (int) Math.floor((a[i]-min)/delta);
			bucketMaxima[bucketIndex] = bucketMaxima[bucketIndex] == Integer.MIN_VALUE ? a[i] : Math.max(bucketMaxima[bucketIndex], a[i]);
			bucketMinima[bucketIndex] = bucketMinima[bucketIndex] == Integer.MAX_VALUE ? a[i] : Math.min(bucketMinima[bucketIndex], a[i]);
		}
		
		//find the maxgap - maxgaps
		int prev =  min;
		int maxGap = 0;
		for(int i = 0; i< n-1; i++){
			//empty bucket according to Pigeonhole principle
			if(bucketMinima[i] == Integer.MAX_VALUE){
				continue;
			}
			
			maxGap = Math.max(maxGap, bucketMinima[i]-prev);
			prev = bucketMaxima[i];
		}
		
		maxGap = Math.max(maxGap, max-prev);
		
		return maxGap;
	}
	
	public static void main(String args[]){
		//5, 3, 1, 8, 9, 2, 4
		//5, 3, 1, 8, 9, 2, 4, 999999,99999
		//1, 2, 1, 1, 1, 1, 4, 2 
		//9, 19, 13, 12, 33, 41, 22
		System.out.println("max gap: "+maxGap(new int[]{5, 3, 1, 8, 9, 2, 4}));
		System.out.println("duplicate within k: "+checkDuplicateWithinK(new int[]{1, 2, 1, 4, 5}, 1));
		System.out.println("duplicate within k mat : "+checkDuplicateWithinK(new int[][]{{1, 2, 3}, {6, 1, 8}}, 2));
		String[][] lists = new String[][]{{"a1","b1","c1","d1"},{"a2","b2","c2"},{"a3","b3","c3"}};
		permuteList(lists, 0, new ArrayList<String>());
		int dec = binaryToDecimal("100001");
		System.out.println("max set bits after single segment bits flipped "+maxSetBitsSingleSegmentFlipped(dec));
		System.out.println("isBstPost : "+isBSTPostOrder(new int[]{1,2,4,3,7,6,5}, 0, 6));
		//(1,11,5), (2,6,7), (3,13,9), (12,7,16), (14,3,25),
        //(19,18,22), (23,13,29), (24,4,28)
//		Building[] bld = new Building[8];
//		bld[0] = new Building(1,11,5);
//		bld[1] = new Building(2,6,7);
//		bld[2] = new Building(3,13,9);
//		bld[3] = new Building (12,7,16);
//		bld[4] = new Building(14,3,25);
//		bld[5] = new Building(19,18,22);
//		bld[6] = new Building(23,13,29);
//		bld[7] = new Building(24,4,28);
		Building[] bld = new Building[3];
		bld[0] = new Building(2,7,4);
		bld[1] = new Building(2,5,4);
		bld[2] = new Building(2,6,4);
		
		ArrayList<Strip> strips = skyLine(bld);
		System.out.println("skyline: "+strips.toString());
		printFactors(12);
		System.out.println("nth ugly : "+nthUglyNumber(1600)+" dp: "+nthUglyDP(1600));
		int[][] following = new int[5][5];
		following[0][3] = 1;
		following[0][1] = 1;
		following[1][3] = 1;
		following[1][0] = 1;
		following[1][4] = 1;
		following[4][3] = 1;
		following[2][3] = 1;
		following[2][4] = 1;
		
		System.out.println("Influencer : "+ influencer(following));
		
		System.out.println("max prod subarray : "+maxProductSubArr(new int[]{1,2,-1,0,2,3,-2,-3,2}));
		System.out.println("total triangles: "+countTriangleTriplets(new int[]{10, 21, 22, 100, 101, 200, 300}));
		String patr = "-?\\d+(\\.\\d+)?";
		System.out.println("is number "+"-23.34".split(patr).length);
		Job[] loadJobs = new Job[5];
		loadJobs[0] = new Job(3, 7, 4);
		loadJobs[1] = new Job(1, 3, 6);
		loadJobs[2] = new Job(4, 9, 5);
		loadJobs[3] = new Job(10, 11, 11);
		loadJobs[4] = new Job(3, 4, 2);
		System.out.println("max load; "+maxLoad(loadJobs));
		String[] exprTokens = {"3", "+", "4", "*", "2", "/", "(", "1", "-", "5", ")", "*", "2", "/", "3"};
		System.out.println("infix: "+convertInfix(exprTokens));
		System.out.println("infix eval : "+evaluateInfix(convertInfix(exprTokens)));
		boolean bb = isOperator("f");
		TreeNode r1 = new TreeNode(1, 1);
		TreeNode r2 = new TreeNode(2, 2);
		TreeNode r3 = new TreeNode(3, 3);
		TreeNode r4 = new TreeNode(4, 4);
		TreeNode r5 = new TreeNode(5, 5);
		
		r1.left = r2;
		r1.right = r3;
		r3.left = r4;
		r3.right = r5;
		
		Codec codec = new Codec();
		TreeNode deser = codec.deserialize(codec.serialize(r1));
		
		
		String rearrng = "aa";
		System.out.println("rearranged: "+rearrangeAdjacentDuplicates(rearrng));
		int[] wig = {3,5,2,1,6,4};
		wiggleSort(wig);
		System.out.println("wiggle sort: "+Arrays.toString(wig));
		MovingAvgLastN mv = new MovingAvgLastN(3);
		mv.add(2);
		System.out.println("mov avg: "+mv.getAvg());
		mv.add(3);
		System.out.println("mov avg: "+mv.getAvg());
		mv.add(4);
		System.out.println("mov avg: "+mv.getAvg());
		mv.add(1);
		System.out.println("mov avg: "+mv.getAvg());
		mv.add(2);
		System.out.println("mov avg: "+mv.getAvg());
		mv.add(-3);
		System.out.println("mov avg: "+mv.getAvg());
		mv.add(0);
		System.out.println("mov avg: "+mv.getAvg());
		mv.add(3);
		System.out.println("mov avg: "+mv.getAvg());
		mv.add(-3);
		System.out.println("mov avg: "+mv.getAvg());
		
		
		int rot[] = {5,6,8,9,1,1,2,3,4,4,4,5};
		System.out.println("search in rot array: "+searchInSortedRotatedArray(rot, 4));
		System.out.println("search rot pos: "+searchRotationPosition(rot));
		Point[] points = new Point[6];
		points[0] = new Point(2, 3);
		points[1] = new Point(-2, 1);
		points[2] = new Point(0, 1);
		points[3] = new Point(-1, 0);
		points[4] = new Point(4, 0);
		points[5] = new Point(2, 2);
		
		System.out.println("k closest points: "+Arrays.toString(closestk(points, 3)));
		
		int maxssnonc[] = {-2, 1, -3, 4, -1, 2, 1, 3, -5, 4};
		System.out.println("max sum subarray non cont : "+maxSumSubSeqNonContagious(maxssnonc));
		minDiffElements(new int[]{0,-6,4,6,5,-2}, new int[]{-4,8,2,3,10,9});
		int ra[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27};
		rotateRight(ra, 38);
		System.out.println("rotate right : "+Arrays.toString(ra));
		int la[] = {1,2,3,4,5,6};
		rotateLeft(la, 2);
		System.out.println("rotate left : "+Arrays.toString(la));
		
		long s11 = System.currentTimeMillis();
		int psm = perfectSquaresLagrange(1000000);
		long e11 = System.currentTimeMillis();
		System.out.println("perfect squares math: "+psm+" time : "+(e11-s11)+" ms");
		s11 = System.currentTimeMillis();
		int psdp = perfectSquareDP(1000000);
		e11 = System.currentTimeMillis();
		System.out.println(" perfect sum dp: "+psdp+" time : "+(e11-s11)+" ms");
		System.out.println("longest valid parenth : "+ longestValidParenthesis2("())()()(()"));
		int[] stock = {23171, 21015, 21123, 21366, 21013, 21367};
		System.out.println("max profit: "+maxStockProfit(stock));
		System.out.println("lexico smaller no dup: "+ lexicoSmallNoDuplicates("bcabc"));
		ArrayList<ArrayList<Integer>> adjList = new ArrayList<ArrayList<Integer>>();
		adjList.add(0, new ArrayList<Integer>());
		adjList.add(1, new ArrayList<Integer>());
		adjList.add(2, new ArrayList<Integer>());
		adjList.add(3, new ArrayList<Integer>());
		adjList.add(4, new ArrayList<Integer>());
		adjList.add(5, new ArrayList<Integer>());
		
		adjList.get(5).add(0);
		adjList.get(5).add(2);
		adjList.get(4).add(0);
		adjList.get(4).add(1);
		adjList.get(2).add(3);
		adjList.get(3).add(1);
		adjList.get(1).add(2);
		
		topologicalSort(adjList);
		
		Set<String> dictn =  new HashSet<String>();
		dictn.add("i");
		dictn.add("like");
		dictn.add("ili");
		dictn.add("kesa");
		dictn.add("msun");
		dictn.add("sam");
		dictn.add("sung");
		//dictn.add("samsung");
		dictn.add("mobile");
		dictn.add("oal");
		dictn.add("goal");
		
		String quer = "ilikesamsungoal";
		ArrayList<String> rest = new ArrayList<String>();
		System.out.println("wordbreak: "+wordBreak(dictn, quer, rest)+" DP: "+wordBreakDP(dictn, quer));
		System.out.println(rest.toString());
		ArrayList<String> allBreaks = wordBreakAll(dictn, quer);
		System.out.println("All breaks: "+allBreaks.toString());
		String stre = "aaa";
		String patt = "ab*";
		System.out.println("patt: "+isMatch(stre, patt)+" --> expected: "+stre.matches(patt));
		printIPAddressedd("");
		int[] maxsumsub = {1, 2, -4, 1, 3, -2, 3, -1};
		System.out.println("Max sum subseq : "+maxSumSubArray(maxsumsub));
		int sa[] = {10, 15, 25};
		int sb[] = {1, 5, 20, 30};
		mergeSortedAlternate(sa, sb, new ArrayList<Integer>(), new int[]{0, 0});
		int pm[] = {3,2,1};
		nextPermutation(pm);
		System.out.println("next perm: "+Arrays.toString(pm));
		ListNode lnode = new ListNode(6);
		lnode.next = new ListNode(7);
		lnode.next.next = new ListNode(8);
		lnode.next.next.next = new ListNode(9);
		lnode.next.next.next.next = new ListNode(10);
		lnode.next.next.next.next.next = new ListNode(13);
		lnode.next.next.next.next.next.next = new ListNode(15);
		lnode.next.next.next.next.next.next.next = new ListNode(16);
		
		//clone root
		ListNode lnodet = new ListNode(-1);
		lnodet.val = lnode.val;
		lnodet.next = lnode.next;
		
		BTNode convertedBTNode2 = convertList2BTreeRecursive(lnodet);
		System.out.println("convertSLL2BTRecursive : ");
		convertedBTNode2.print(convertedBTNode2);
		System.out.println();
		ListNode convertedBTNode1 = convertList2BTreeInplace(lnode);
		System.out.println("convertSLL2BTInplace : ");
		convertedBTNode1.printAsTree(convertedBTNode1);
		System.out.println();
		
		MergeSortList(lnode);
		
		ListNode deleteNode = deleteNodeWithHighOnRight(lnode);
		ListNode reversed = reverse(lnode,3);
		
		ListNode split2 = splitLinkedListNode2(reversed, 2);
		//ListNode split = splitLinkedListNode(lnode, 2);
		
		String allCase = "aBcd";
		Set<String> resAlCases = new HashSet<String>();
		allCasePermutataion(allCase, 0, resAlCases);
		System.out.println("all cases : "+resAlCases.toString());
		
		BTNode bt1 = new BTNode(1);
		BTNode bt2 = new BTNode(2);
		BTNode bt3 = new BTNode(3);
		BTNode bt4 = new BTNode(4);
		BTNode bt5 = new BTNode(5);
		BTNode bt6 = new BTNode(6);
		BTNode bt7 = new BTNode(7);
		BTNode bt8 = new BTNode(8);
		
		bt1.left = bt2;
		bt1.right = bt3;
		bt2.left = bt4;
		bt3.left = bt5;
		bt3.right = bt6;
		bt4.right = bt7;
		bt6.left = bt8;
		
		//BTNode headdd = inorderInPlaceUsingStack(bt1);
		
		int max[] = {0};
		maxSumPath(bt1, max);
		System.out.println("maxSum path : "+max[0]);
		int max1[] = {0};
		maxSumPath1(bt1, max1);
		System.out.println("maxSum path1 : "+max1[0]);
		
		int[] lens = new int[]{0, Integer.MAX_VALUE};
		BTNode closest = new BTNode(-1);
		closest = closestLeaf(bt1, lens, closest);
		
		System.out.println("stack inorder traversal : ");
		InorderTraversal(bt1);
		
		System.out.println("morris inorder traversal : ");
		MorrisInorderTraversal(bt1);
		
		ListNode sll = convertToList(bt1);
		//BTNode sllip = convertToListInplace(bt1);
		
		ListNode llNode = flattenBT2SLL(bt1);
		LinkedList<BTNode> llNFast = new LinkedList<test.BTNode>();
		flattenBT2SLLFaster(bt1, llNFast);
		BTNode convertedBTNode = convertList2BTreeIterative(llNode);
		BTNode rtMostCousin = rightMostCousin2(bt1, 7);
		if(rtMostCousin != null){
			System.out.println("right most cousin : "+rtMostCousin.toString());
		}
		connectLevelOrder(bt1);
		System.out.println("level order: ");
		printLevelOrder(bt1);
		
		mirrorTree(bt1);
		
		System.out.println("diameter: "+diameter(bt1)+" diam 2: "+diameter(bt1, new int[]{0}));
		System.out.println("level : "+getLevel(bt1, 1, bt7));
		System.out.println("level : "+getLevel(bt1, 1, bt8));
		System.out.println("level : "+getLevel(bt1, 1, bt1));
		System.out.println("max depth : "+maxDepth(bt1));
		
		System.out.println("shortest path: "+shortestDistance(bt1, bt5, bt8));
		System.out.println("mirror Tree : ");
		
		String path = "";
		ArrayList<String> paths = new ArrayList<String>();
		paths(bt1, path, paths);
		System.out.println("all paths : "+paths.toString());
		
		
		BTNode nnnn11 = new BTNode(11);
		BTNode nnnn5 = new BTNode(5);
		BTNode nnnn15 = new BTNode(15);
		BTNode nnnn3 = new BTNode(3);
		BTNode nnnn9 = new BTNode(9);
		BTNode nnnn12 = new BTNode(12);
		BTNode nnnn20 = new BTNode(20);
		
		nnnn11.left = nnnn5;
		nnnn11.right = nnnn15;
		nnnn5.left = nnnn3;
		nnnn5.right = nnnn9;
		nnnn15.left = nnnn12;
		nnnn15.right = nnnn20;
		
		BTNode sortedDLL = flattenBST2SortedCircularDLLInplace(nnnn11);
		 
		System.out.println("first unique : "+firstUnique("acadac".toCharArray()));
		int[] perm = {3, 1, 2};
		System.out.println("perm rank: "+permRank(perm));;
		int slidw[] = {2,1,3,4,6,3,8,9,10,12,56};
		System.out.println("sliding max : "+Arrays.toString(slidingWindowMax(slidw, 4)));
		System.out.println("sliding min : "+Arrays.toString(slidingWindowMin(slidw, 4)));
		Dictionary dict = new Dictionary();
		dict.add("aab");
		dict.add("aaaa");
		dict.add("baa");
		dict.add("abc");
		dict.add("bbbca");
		dict.add("aba");
		dict.add("abb");
		dict.add("bbb");
		
		int cnnt = dict.searchAnagram("aab");
		System.out.println("anagrams in dict : "+cnnt);
		
		String text = "abateatas";
		String pat = "tea";
		int cnt = searchAnagramSubstring(text, pat);
		System.out.println("total anamgrams: "+cnt);;
		String strp = "dabbae";
		System.out.println("longest palind: "+longestPalindrom(strp));
		System.out.println("longest palind min insertions: "+	minInsertionsForLongestPalindrom(strp));
		int maxA[] = {2, 0 ,3 ,2 ,1 };
		System.out.println("max area: "+maxArea(maxA));
		Job[] jobs = {new Job(1,2,50),new Job(3,5,20),new Job(6,19,100),new Job(2,100,200)};
		System.out.println("activity dselection max profit : "+weightedActivitySelection(jobs));
		int sorted[] = {2,5,19,100};
		int key = 2;
		System.out.println("floor: "+binarySearchFloor(sorted, 0, sorted.length-1, key)+" Ceil: "+binarySearchCeiling(sorted, 0, sorted.length-1, key));
		int dnf[] = {1,0,1,2,2,0,0,1,1,2};
		DNFSort(dnf);
		
		int aa[] = {1, 2, 6, 100, 104, 1, 2, 6};		
		int as = 1;
		int ae = 5;
		System.out.println("count : "+subArrayWithSumInRange(aa, as, ae));
		System.out.println("count1 : "+subArrayWithSumInRange1(aa, as, ae));
		
		int[] AA = {2,3,4};
		int[] BB = {1};
//		System.out.println("kth smallest :");
//		for(int i = 1; i<16; i++){
//			System.out.print(findKthSmallest(AA, 0, AA.length-1, AA.length, BB, 0, BB.length-1, BB.length, i));
//		}
		System.out.println("median : "+findMedianSortedArrays(AA, BB));
		System.out.println("median1 : "+findMedianSortedArrays1(AA, BB));
		int INF = Integer.MAX_VALUE;
		int [][] room = {{INF, -1, 0, INF},{INF, INF, INF, -1},{INF, -1 , INF, -1},{0, -1, INF, INF}};
		findExitWallsAndGates(room);
		int dups[] = {1, 2, 3, 5, 2};
		System.out.println("duplicate: "+findDuplicate(dups));
		int icA[] = {2, 7, 5, 5, 2, 7, 0, 8, 1};
		//[3, 6, 4, 3, 2, 2, 0, 1, 0]
		System.out.println("count smaller on right : "+Arrays.toString(countSmallerOnRight(icA)));
		System.out.println("count smaller on right using inv count: "+Arrays.toString(countSmallerOnRightWithMerge(icA)));
		
//		int rank[] = new int[icA.length];
//		for(int i=0;i<icA.length;i++){
//			rank[i] = i;
//		}
//		int ic[] = new int[icA.length];
//		int icc = inversionCountGen(icA, rank, 0, icA.length-1, ic);
//		System.out.println("inv count: "+icc+"+ inversions: "+Arrays.toString(ic));
		int iccca[] = {2 ,4, 1, 3, 5};
		System.out.println("inv count: " + mergeSortWithInvCount(iccca, 0, iccca.length-1));
		
		TreeNode nnn11 = new TreeNode(11);
		TreeNode nnn5 = new TreeNode(5);
		TreeNode nnn15 = new TreeNode(15);
		TreeNode nnn3 = new TreeNode(3);
		TreeNode nnn9 = new TreeNode(9);
		TreeNode nnn12 = new TreeNode(12);
		TreeNode nnn20 = new TreeNode(20);
		TreeNode nnn7 = new TreeNode(8);
		TreeNode nnn10 = new TreeNode(10);
		
		nnn11.left = nnn5;
		nnn11.right = nnn15;
		nnn5.left = nnn3;
		nnn5.right = nnn9;
		nnn9.left = nnn7;
		nnn9.right = nnn10;
		nnn15.left = nnn12;
		nnn15.right = nnn20;
		
		int kk = 18;
		System.out.println("closest to "+kk+" is: "+findCLosestBST(nnn11, kk, Integer.MAX_VALUE, Integer.MAX_VALUE));
		
		int mat[][] = {{2,4,5,6},{1,2,2,4},{3,4,4,5},{1,2,3,3}};
		
		for(int i=0; i<16; i++){
			System.out.println(kthSmallestElement(mat, i+1));
		}
		
		int A2[] = {1, 2, 3, 10, 15};
		int B2[] = {-1, 3, 6, 7};
		
		//System.out.println("kth smallest from two sorted arrays: "+kthSmallestElement(A2, 0, A2.length-1, B2, 0, B2.length-1, 6));
		System.out.println("median of two sorted arrays: "+median(A2, 0, A2.length-1, B2, 0, B2.length-1));
		
		TreeNode n1 = new TreeNode(1);
		TreeNode n2 = new TreeNode(2);
		TreeNode n3 = new TreeNode(3);
		TreeNode n4 = new TreeNode(4);
		TreeNode n5 = new TreeNode(5);
		TreeNode n6 = new TreeNode(6);
		TreeNode n7 = new TreeNode(7);
		TreeNode n8 = new TreeNode(8);
		TreeNode n9 = new TreeNode(9);
		n1.left = n2;
		n1.right = n3;
		n2.left = n4;
		n2.right = n5;
		n5.left = n6;
		n5.right = n7;
		n6.left = n8;
		n6.right = n9;
		
		TreeNode[] sampleK = randomKSampleTreeNode(n1, 3);
		
		for(int i = 0; i<sampleK.length; i++){
			System.out.print("--> "+sampleK[i].key);
		}
		System.out.println();
		
		String str1 = "abc";
		permutation(str1, 3, false);
		permutation(str1, 3, true);
		String str2 = "1123";
		uniquePermutation(str2, 3);
		
		combination(str1, 2);
		allCombination(str1);
		
		SerializableTree root = new SerializableTree("A");
		SerializableTree b = new SerializableTree("rank");
		SerializableTree c = new SerializableTree("surp");
		SerializableTree d = new SerializableTree("d");
		SerializableTree e = new SerializableTree("e");
		SerializableTree f = new SerializableTree("f");
		SerializableTree g = new SerializableTree("g");
		c.addChild(e);c.addChild(f);
		d.addChild(g);
		root.addChild(b);
		root.addChild(c);
		root.addChild(d);
		
		String serTree = SerializableTree.serialize(root);
		System.out.println("--> serialzied tree: "+serTree);
		SerializableTree deserilizedTree = SerializableTree.deserialize(serTree);
		System.out.println("--> serialzied tree of the deserialized: "+SerializableTree.serialize(deserilizedTree));
		
		int m[][] = {{0,0,1,0,0},{1,1,1,0,1},{1,1,1,1,1},{1,1,1,1,0}};
		System.out.println("minPlusLen: "+largestPlusInMatrix(m));
		
		int A1[] = {1, 2, 3, 4, 5, 6, 7 , 8, 9};
		
		ArrayList<Integer>[] res1 = findEqualPartitionMinSumDif(A1);
		
		int set[] = {1, 3, 5};
		System.out.println(" subset sum : "+ isSubSetSum(set, 7));
		
		int str[] = {1, 2, 5, 3, 7, 6};
		System.out.println(" stream median : "+ getStreamMedian(str));
		
		//double[] steps = {1.0,2.0,2.0,4.0,3.0,3.0,0.5};
		//double steps[] = {2, 1, 4, 4, 3, 3, 2, 1, 1};
		double steps[] = {1, 2, 4, 4, 4, 3};
		System.out.println("isCrossed: "+isCrossed(steps));
		
		int[] A = {1, 3, 4, 6, 2, 3, 5};
		
		test.mergeInPlace(A, 0, 4);
		//test.mergeSortInPlace(A, 0, A.length-1);
		
		int [] B = {1, 5, 2, 3, 6, 4};
		test.mergeSortInPlace(B, 0, B.length-1);
		//test.mergeInPlace(B, 2, B.length-1);
		
		int [] C = {1, 3, 4, 6, 2, 3, 5};
		test.merge(C, 0, 4, C.length-1);
		
		String value = "42.12";
			double dvalue = Double.parseDouble(value);
			int whole = (int)dvalue;
			double fraction = dvalue-whole;
			
			if(fraction <= 0.0){
				value = whole+"";
			}
			
		
		int count = 0;
		int what = 1;
		for(int i=1000 ; i<=2000; i++){
			String s = i+"";
			boolean failed = false;
			
			for(int j=0; j< s.length(); j++){
				if((s.charAt(j)-'0')%2 != what){
					failed = true;
				}
			}
			if(!failed)
				count++;
		}
		
		System.out.println("\ncount = "+count);
		
		 if("fgg_06".matches("^([0-9]|_).*")){
			 System.out.println("found...");
		 }
		 else{
			 System.out.println("not found...");
		 }
		
		int[] digits = new int[]{4,2,5,7,3,1};
		int[] out = test.nextEven(digits);
		
		test t = new test();
		
		String[] res = t.topKWords("a b b c c c e e e e e d d d d g g g g g g g f f f f f f", 3);
		
		System.out.println("");
		
		TreeNode nn6 = new TreeNode(6);
		TreeNode nn7 = new TreeNode(7);
		TreeNode nn117 = new TreeNode(117);
		TreeNode nn4 = new TreeNode(4);
		TreeNode nn3 = new TreeNode(3);
		TreeNode nn5 = new TreeNode(5);
		
		nn6.left = nn4;
		nn6.right = nn117;
		nn4.left = nn3;
		nn4.right = nn5;
		
		int len = 0;
		int rlen = t.minLenSumPathBST(nn6, 13, len);
		
		System.out.println(rlen);
		
		Set<String> laddict = new HashSet<>();
		//laddict.add("hit");
		laddict.add("hot");
		laddict.add("dot");
		laddict.add("dog");
		laddict.add("lot");
		laddict.add("log");
		laddict.add("lob");
		laddict.add("cob");
		//laddict.add("cog");
		//isNavigable("hit", "cog", laddict);
		List<List<String>> resPaths = wordLadderAll(laddict, "hit", "cog");
		System.out.println(resPaths.toString());
	}
}
