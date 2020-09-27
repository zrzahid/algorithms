package test;

import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import test.Test.Graph.Edge;

class Solution {
    int m;
    int n;
    int[] parent;
    int[] size;
    int count;
    
    public int gridToSetIndex(int i, int j){
        return i*m + j;
    }
    
    public int find(int x){
        if(parent[x] == x){
            return x;
        }
        else{
            return find(parent[x]);
        }
    }
    
    public void union(int x, int y){
        int rootX = find(x);
        int rootY = find(y);
        
        if(rootX == rootY)
            return;
        else
            count--;
        
        if(size[rootX] >= size[rootY]){
            size[rootX] += size[rootY];
            parent[rootY] = rootX;
        }
        else{
            size[rootY] += size[rootX];
            parent[rootX] = rootY;
        }
    }
    
    public boolean isSameIsland(int x, int y){
        return find(x) == find(y);
    }
    
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0)  {
            return 0;  
        }
        
        m = grid.length;
        n = grid[0].length;
        parent = new int[m*n];
        size = new int[m*n];
        count = 0;
        
        for(int i = 0; i < parent.length; i++){
            parent[i] = i;
            size[i] = 1;
            count++;
        }
        
        int[][] neighbors = {{1,0},{-1,0},{0,1},{0,-1}};
        int islands = 0;
        for(int i = 0; i< m; i++){
            for (int j = 0; j < n; j++){
                int x = gridToSetIndex(i, j);
                
                // if land then test if neighbor grids are also part of same land
                if(grid[i][j] == '1'){
                    for(int[] nbr : neighbors){
                        int y = gridToSetIndex(i+nbr[0], j+nbr[1]);

                        if (x >= 0 && x < m && y >= 0 && y < n && grid[x][y] == '1') {
                            union(x, y);
                        }
                    }
                }
            }
        }
        
        return count;
    }
}

public class Test {
    
    class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {}

        public Node(int _val,Node _left,Node _right,Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    };
    
    public Node connect(Node root) {
        // visit each level and connect it;s childrens
        if(root ==  null) {
            return null;
        }
        
        Node cur = null;
        Node preRoot = root;
        while(preRoot.left != null) {
            
            cur = preRoot;
            
            while(cur != null) {
                cur.left.next = cur.right;
                if(cur.next != null) {
                    cur.right.next = cur.next.left;
                }
                cur = cur.next;
            }

            preRoot = preRoot.left;
        }
        
        return root;
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        List<Integer> res = new LinkedList<>();
        
        while(root != null || !stack.isEmpty()) {
            // go far lett (min node) and stack up all nodes
            while(root != null) {
                stack.push(root);
                root = root.left;
            }
            
            // now pop the min node
            root = stack.pop();
            // add to the result
            res.add(root.val);
            // as there was no left subtree of this min node go to right
            root = root.right;
        }
        
        return res;
    }
    
    //DLL
    public TreeNode inorderDLListInplace(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode head = null;
        TreeNode curHead = null;
        
        while(root != null || !stack.isEmpty()) {
            // go far lett (min node) and stack up all nodes
            while(root != null) {
                stack.push(root);
                root = root.left;
            }
            
            // now pop the min node
            root = stack.pop();
            // use right pointer for link
            if(head == null) {
                head = root;
                curHead = head;
            }
            else {
                // DLL
                curHead.right = root;
                root.left = curHead;
                
                curHead = curHead.right;
            }
            // as there was no left subtree of this min node go to right
            root = root.right;
        }
        
        head.left = null;
        return head;
    }
    
    public TreeNode connectDLLs(TreeNode node1, TreeNode node2) {
        
        if(node1 == null || node2 == null) {
            return node2;
        }
        
        TreeNode tail1 = node1.left;
        TreeNode tail2 = node2.left;
        
        // connect tail1 to node2
        tail1.right = node2;
        node2.left = tail1;
        // connect tail2 to node1
        tail2.right = node1;
        node1.left = tail2;
        
        return node1;
    }
    
    public TreeNode inorderCircularDLListInplace(TreeNode root) { 
        
        if(root == null) {
            return null;
        }
        
        TreeNode left = inorderCircularDLListInplace(root.left);
        TreeNode right = inorderCircularDLListInplace(root.right);
        
        // make the node a circular dll
        root.left = root;
        root.right = root;
        
        // connect left circular dll to root circular dll
        connectDLLs(left, root);
        // connect new left circular dll to right circular dll
        connectDLLs(left, right);
        
        return left;
        
    }
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        Deque<TreeNode> stack = new ArrayDeque<>();
        TreeNode p = root;
        while(!stack.isEmpty() || p != null) {
            if(p != null) {
                stack.push(p);
                result.add(p.val);  // Add before going to children
                p = p.left;
            } else {
                TreeNode node = stack.pop();
                p = node.right;   
            }
        }
        return result;
    }
    
    public List<Integer> postorderTraversal(TreeNode root) {
        LinkedList<Integer> result = new LinkedList<>();
        Deque<TreeNode> stack = new ArrayDeque<>();
        TreeNode p = root;
        while(!stack.isEmpty() || p != null) {
            if(p != null) {
                stack.push(p);
                result.addFirst(p.val);  // Reverse the process of preorder
                p = p.right;             // Reverse the process of preorder
            } else {
                TreeNode node = stack.pop();
                p = node.left;           // Reverse the process of preorder
            }
        }
        return result;
    }
    
    public void inorderMorris(TreeNode root){
        
        TreeNode cur = root;
        TreeNode pre = null;
        
        while(cur != null) {
            // if no left child then print current and go to right subtree
            if(cur.left == null) {
                System.out.print(cur.val+ " ");
                cur = cur.right;
            }
            else {
                // left subtree exists. Inorder traversal has to come back to this current node after the left subtree is traversed
                // usually we can achieve it by pushing the cur node to stack. 
                // but without any stack how do we achieve it?
                // we know that the last traversed node in the left subtree will be 
                // the left subtree, which is the predecessor for current node. 
                // So, we can make the traverse come back to current node naturally 
                // by threading the right most child's right pointer to the current node.
                
                // so first find the predecessor of cur and thread it's right pointer to current
                // If the threaded pointer was set already then we have naturally 
                // came back to the root through the threaded pointer. So, not need to thread agains. 
                pre = cur.left;
                while(pre.right != null && pre.right != cur) {
                    pre = pre.right;
                }
                
                // if the predecessor is found than thread it's right pointer to current
                // and then traverse to next left
                if(pre.right == null) {
                    pre.right = cur;
                    cur = cur.left;
                }
                // if threaded pointer was already set in previous step than we have naturally 
                // came back to the root through the threaded pointer. 
                // so, print the root, unthead the predecessor, and go to right
                else {
                    System.out.print(cur.val+ " ");
                    pre.right = null;
                    cur = cur.right;
                }
            }
        }
    }
    
    public void preMorris(TreeNode root){
        
        TreeNode cur = root;
        TreeNode pre = null;
        
        while(cur != null) {
            // if no left child then print current and go to right subtree
            if(cur.left == null) {
                System.out.print(cur.val+ " ");
                cur = cur.right;
            }
            else {
                // left subtree exists. Inorder traversal has to come back to this current node after the left subtree is traversed
                // usually we can achieve it by pushing the cur node to stack. 
                // but without any stack how do we achieve it?
                // we know that the last traversed node in the left subtree will be 
                // the left subtree, which is the predecessor for current node. 
                // So, we can make the traverse come back to current node naturally 
                // by threading the right most child's right pointer to the current node.
                
                // so first find the predecessor of cur and thread it's right pointer to current
                // If the threaded pointer was set already then we have naturally 
                // came back to the root through the threaded pointer. So, not need to thread agains. 
                pre = cur.left;
                while(pre.right != null && pre.right != cur) {
                    pre = pre.right;
                }
                
                // if the predecessor is found than thread it's right pointer to current
                // and then traverse to next left
                if(pre.right == null) {
                    pre.right = cur;
                    System.out.print(cur.val+ " "); // preorder - print as soon as we traversea node
                    cur = cur.left;
                }
                // if threaded pointer was already set in previous step than we have naturally 
                // came back to the root through the threaded pointer. 
                // so, print the root, unthead the predecessor, and go to right
                else {
                    pre.right = null;
                    cur = cur.right;
                }
            }
        }
    }
    
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }

    }

    private int[] preorder;
    private int[] inorder;

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        this.preorder = preorder;
        this.inorder = inorder;
        return buildTreeHelper(0, inorder.length - 1, 0);
    }

    public int findRootIndex(int istart, int iend, int nextPre){
        for(int i = istart; i <= iend; i++){
            if(this.preorder[nextPre] == this.inorder[i]){
                return i;
            }
        }   
        
        return -1;
    }

    public TreeNode buildTreeHelper(int istart, int iend, int nextPre) {
        if (istart < iend || nextPre >= this.preorder.length - 1) {
            return null;
        }

        int rootIndex = findRootIndex(istart, iend, nextPre);
        if (rootIndex == -1) {
            return null;
        }

        TreeNode root = new TreeNode(this.inorder[rootIndex]);
        root.left = buildTreeHelper(istart, rootIndex - 1, nextPre + 1);
        root.right = buildTreeHelper(rootIndex + 1, iend, nextPre + (rootIndex - istart) + 1);

        return root;
    }

    public class ListNode {
        int val;
        ListNode next;
        ListNode prev;

        public ListNode(int x) {
            val = x;
        }
    }

    public ListNode oddEvenList(ListNode head) {
        if (head == null || head.next == null || head.next.next == null) {
            return head;
        }

        ListNode oddHead = head;// 1->2...
        ListNode oddTail = head;// 1->2...
        ListNode evenHead = head.next;// 2->3...
        ListNode evenTail = head.next;// 2->3...

        boolean isOdd = true;
        head = head.next.next;// 3->4...
        while (head != null) {// 3
            if (isOdd) {// true
                oddTail.next = head;// 1->3->4..
                oddTail = oddTail.next;// 3->4..
            } else {
                evenTail.next = head;// 2->4
                evenTail = evenTail.next;
            }

            isOdd = !isOdd;// false
            head = head.next;// 4->...
        }

        evenTail.next = null;
        oddTail.next = evenHead;
        return oddHead;
    }
    
    /**
     * 
     *    3
         / \
        9  20
          /  \
         15   7
     * @param root
     * 
     * preorder: 3, 9, 20, 15, 7
     */
    public void preorder(TreeNode root, boolean reverse) {
        if(root == null) {
            return;
        }
        
        System.out.println(root.val);
        if(reverse) {
            preorder(root.right, !reverse);
            preorder(root.left, !reverse);
        }
        else {
            preorder(root.left, !reverse);
            preorder(root.right, !reverse);
        }
    }
    
    public void permList(List<List<String>> input, String[] cur, int index, List<List<String>> result) {
        if(index == input.size()) {
            result.add(Arrays.asList(cur.clone()));
        }
        else {
            // for each candidate in current position (index) recurse to construct solution
            List<String> cands = input.get(index);
            for(int i = 0; i< cands.size(); i++) {
                // add current candidate to temp solution
                cur[index] = cands.get(i);
                permList(input, cur, index+1, result);
                //backtrack
                cur[index] = null;
            }
        }
    }
    
    public String[] map = new String[] {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    public List<String> letterCombinations(String digits) {
        List<String> result = new ArrayList<>();
        if(digits == null || digits.isEmpty()){
            return result;
        }
        permList(digits, new StringBuilder(), 0, result);
        return result;
    }
    
    public void permList(String input, StringBuilder cur, int index, List<String> result) {
        if(index == input.length()) {
            result.add(cur.toString());
        }
        else {
            // for each candidate in current position (index) recurse to construct solution
            String cands = map[input.charAt(index)-'0'];
            for(int i = 0; i< cands.length(); i++) {
                // add current candidate to temp solution
                cur.append(cands.charAt(i));
                permList(input, cur, index+1, result);
                //backtrack
                cur.setLength(cur.length() - 1);
            }
        }
    }
    
    public class Graph{
        class Edge{
            public int u;
            public int v;
            public int w;
            
            public Edge(int u, int v, int w) {
                this.u = u;
                this.v = v;
                this.w = w;
            }
        }
        
        int[] vertices;
        Edge[][] edges;
        
        
        public List<Integer> shortestPath(int start, int end) {
            List<Integer> shortestPath = new LinkedList<>();
            int[] shortestPathLen = new int[vertices.length+1];
            Arrays.fill(shortestPathLen, Integer.MAX_VALUE);
            
            int[] shortestPathParent = new int[vertices.length+1];
            Arrays.fill(shortestPathParent, -1);
            
            boolean[] visited = new boolean[vertices.length+1];
            Queue<Integer> queue = new LinkedList<>();
            queue.add(start);
            
            while(!queue.isEmpty()) {
                int u = queue.remove();
                visited[u] = true;
                
                for(Edge e : edges[u]) {
                    if(e != null) {
                        if((shortestPathLen[u] + e.w) < shortestPathLen[e.v]) {
                            shortestPathLen[e.v] = shortestPathLen[u] + e.w;
                            shortestPathParent[e.v] = u;
                        }
                        
                        if(!visited[e.v]) {
                            queue.add(e.v);
                        }
                    }
                }
            }
            
            int i = end;
            while(shortestPathParent[i] != -1) {
                shortestPath.add(0, vertices[i]);
                i = shortestPathParent[i];
            }
            shortestPath.add(0, vertices[start]);
            
            return shortestPath;
        }
    }
    
    public List<int[]> getWalkCandidates(int[][] m, int i, int j, boolean[][] visited){
        int[][] dirs = new int[][] {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        List<int[]> cands = new ArrayList<>();
        for(int[] dir : dirs) {
            int k = i+dir[0];
            int l = j+dir[1];
            
            if(k >= m.length || l >= m[0].length || k < 0 || l < 0) {
                continue;
            }
            if(!visited[k][l] && canWalk(m, i, j, k, l)) {
                cands.add(new int[] {k, l});
            }
        }
        
        return cands;
    }
    public boolean canWalk(int[][] m, int i, int j, int k, int l) {
        return m[i][j] < m[k][l];
    }
    Stack<Integer> maxPath = new Stack<Integer>();
    public void walkDFS(int[][] m, Stack<Integer> path, int i, int j, boolean[][] visited) {
        path.push(m[i][j]);
        visited[i][j] = true;
        if(path.size() > maxPath.size()) {
            maxPath = (Stack<Integer>) path.clone();
        }
        
        List<int[]> cands = getWalkCandidates(m, i, j, visited);
        if(cands.isEmpty()) {
            path.pop();
            return;
        }
        
        for(int[] cand : cands) {
            walkDFS(m, path, cand[0], cand[1], visited);
        }
        
        path.pop();
    }
    
    public List<Integer> walkDFS(int[][] m) {
        for(int i = 0; i < m.length; i++) {
            for(int j = 0; j< m[0].length; j++) {
                boolean[][] visited = new boolean[m.length][m[0].length];
                Stack<Integer> path = new Stack<Integer>();
                walkDFS(m, path, i, j, visited);
            }
        }
        
        return maxPath;
    }
    
    public int match(int[] text, int[] patt) {
        int count = 0;
        for(int i = 0; i < patt.length; i++) {
            if(patt[i] != 0 && text[i] != 0) {
                count++;
            }
        }
        
        return count;
    }
    
    public String minLenSuperSubString(String s, String t) {
        
        int[] histS = new int[26];
        Arrays.fill(histS, 0);
        int[] histT = new int[26];
        Arrays.fill(histT, 0);
        
        for(char c : t.toCharArray()) {
            histT[c-'a']++;
        }
        
        int start = 0, bestStart = 0, len = 0, minLen = Integer.MAX_VALUE;
        
        int j = start;
        while(start < s.length()) {
            int matchCount = match(histS, histT);
            
            // increase the window forward as long as we don't match all the 
            // chars in t at least once
            while(j < s.length() && matchCount < t.length()) {
                histS[s.charAt(j)-'a']++;
                matchCount = match(histS, histT);
                j++;
            }
            
            // no solution from this start position
            if(matchCount < t.length()) {
                break;
            }
            // we found a substring
            else{
                len = j - start;
                if(len < minLen) {
                    minLen = len;
                    bestStart = start;
                }
            }
            
            // try to shrink the window
            histS[s.charAt(start)-'a']--;
            start++;
        }
        
        return s.substring(bestStart, bestStart+minLen);
    }
    
    public int[] searchRange(int[] nums, int target) {
        int l = 0;
        int h = nums.length - 1;
        int start = -1;
        int end = -1;
        
        while(l < h){
            int mid = l + (h-l)/2;
            
            if(nums[mid] == target){
                end = Math.max(end, mid);
                l = mid+1;
            }
            else if(nums[mid] < target){
                l = mid+1;
            }
            else{
                h = mid;
            }
        }
        
        return new int[]{start, end};
    }
    
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new LinkedList<>();
        Arrays.sort(nums);
        
        for(int k = 0; k < nums.length-2; k++){
            if(k == 0 || (k > 0 && nums[k] != nums[k-1])){
                int sumKey = 0-nums[k];
                int i = k+1, j = nums.length-1;
                while(i < j){
                    if((nums[i]+nums[j]) > sumKey){
                        j--;
                    }
                    else if((nums[i]+nums[j]) < sumKey){
                        i++;
                    }
                    else{         
                        result.add(Arrays.asList(nums[k], nums[i], nums[j]));
                        while(i < j && nums[i] == nums[i+1]) i++;
                        while(i < j && nums[j] == nums[j-1]) j--;
                        i++;
                        j--;
                    }
                }
            }
        }
        
        return result;
    }
    
    public void setZeroes(int[][] matrix) {
        if(matrix.length == 0){
            return;
        }
        int n = matrix.length;
        int m = matrix[0].length;
        boolean[] rows = new boolean[n];
        boolean[] cols = new boolean[m];
        
        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){
                if(matrix[i][j] == 0){
                    rows[i] = true;
                    cols[j] = true;
                }
            }
        }
        
        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){
                if(rows[i] || cols[j]){
                    matrix[i][j] = 0;
                }
            }
        }
    }
    
    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> result = new LinkedList<List<String>>();
        
        Map<Integer, List<String>> histMap = new HashMap<>();
        
        for(String s: strs){
            int hist = hash(s);
            
            List<String> res = histMap.get(hist);
            if(res == null){
                res = new LinkedList<String>();
                histMap.put(hist, res);
                result.add(res);
            }
            
            res.add(s);
        }
        
        return result;
    }
    
    int[] primes = new int[]{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101};
    
    public int hash(String s){
        int hash = 1;
        for(int i = 0; i< s.length(); i++){
            hash *= primes[s.charAt(i) - 'a'];
        }
        
        return hash;
    }
    
    public int lengthOfLongestSubstring(String s) {
        if(s == null || s.isEmpty()){
            return 0;
        }
        
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
    
    class LongestPalindrome {
        int startIndex=0, max=0;
        public String longestPalindrome(String s) {
            if (s.length()<2)
                return s;
            char[] c= s.toCharArray();
            int i=0;
            while ( i < c.length){
                System.out.println(i);
                i = checkPalin(c, i);
            }
            
            return s.substring(startIndex, startIndex + max);
            
        }
        public int checkPalin(char[] c, int i){
            int end = i+1;
            
            while (end < c.length && c[i] == c[end])
                end ++;
            int next = end;
            
            int start = i-1;
            
            while (start>-1 && end < c.length){
                if (c[start] == c[end]){
                    start--;
                    end++;
                }
                else
                    break;
            }
            
            if (end-start-1 > max){
                max = end-start-1;
                startIndex = ++start;
            }
            
            
            return next;
            
        }
    }
    
    public boolean increasingTripletSubseq(int[] a) {
        int[] lis = new int[a.length];
        //base cases - single number is a lis and lds
        Arrays.fill(lis, 1);
        //longest increasing subsequence
        //lis(i) = max{1+lis(j)}, for all j < i and a[j] < a[i]
        for(int i = 1; i < a.length; i++){
            for(int j = 0; j < i; j++){
                if(a[i] > a[j] && lis[j] + 1 > lis[i]){
                    lis[i] = lis[j]+1;
                    if(lis[i] >= 3){
                        return true;
                    }
                }
            }
        }
        
        return false;
    }
    
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode headA1 = headA;
        ListNode headB1 = headB;
        
        while(headA1 != headB1){
            headA1 = headA1 != null? headA1.next : headB;
            headB1 = headB1 != null? headB1.next : headA;
        }
        
        return headA1;
    }
    
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> result = new LinkedList<>();
        
        if(root == null){
            return result;
        }
        
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.add(root);
        int count = 1;
        int level = 0;
        
        LinkedList<Integer> curLevel = new LinkedList<>();
        while(!queue.isEmpty()){
            TreeNode cur = queue.remove();
            count--;
            
            // add to cur level list
            if(level % 2 == 0) 
                curLevel.addLast(cur.val);
            else
                curLevel.addFirst(cur.val);
            
            if(cur.left != null){
                queue.add(cur.left);
            }
            if(cur.right != null){
                queue.add(cur.right);
            }
            
            // end of current level
            if(count == 0){
                level++;
                count = queue.size();
                result.add(curLevel);
                curLevel = new LinkedList<>();
            }
        }
        
        return result;
    }
    
    public int findPeakElement(int[] nums) {
        int l = 0;
        int h = nums.length-1;
        int mid = 0;
        
        while(l < h){
            mid = l + (h-l)/2;
            if(nums[mid] < nums[mid+1]){
                l = mid+1;
            }
            else{
                h = mid;
            }
        }
        
        return l;
    }
    
    public int searchRotated(int[] nums, int target) {
        int n = nums.length;
        int l = 0;
        int h = n-1;
        while(l < h){
            int mid = (l+h)/2;
            
            // rotated part
            if(nums[mid] > nums[h]){
                l = mid+1;
            }
            // sorted part
            else{
                h = mid;
            }
        }
        
        int pivot = l;
        l = 0;
        h = n-1;
        
        while(l <= h){
            int mid = (l+h)/2;
            int actualmid = (mid+pivot)%n;
            
            if(nums[actualmid] == target){
                return actualmid;
            }
            else if(nums[actualmid] < target){
                l = mid + 1;
            }
            else{
                 h = mid - 1;
            }
        }
        
        return -1;
    }
    
    public class Interval{
        public int start;
        public int end;
        public int height;
        
        public Interval(int x, int y) {
            start = x;
            end = y;
        }
        
        public Interval(int x, int z, int y) {
            start = x;
            end = y;
            height = z;
        }
    }
    
    public int[][] mergeOverlappedIntervals(int[][] intervals) {
        if(intervals.length == 0){
            return new int[0][0];
        }
        
        // O(nlgn) sort
        Arrays.sort(intervals, (a,b) -> a[0]-b[0]);
        
        LinkedHashSet<int[]> res = new LinkedHashSet<>();
        int[] prev = intervals[0];
        
        // O(n) for overlaps
        for(int i = 1; i < intervals.length; i++) {
            // overlaps
            if(intervals[i][0] <= prev[1]) {
                // merge intervals
                prev[1] = Math.max( prev[1], intervals[i][1]);
            }
            // current interval ends
            else {
                res.add(prev);
                prev = intervals[i];
            }
        }
        
        if(!res.contains(prev)) {
            res.add(prev);
        }
        
        return res.toArray(new int[res.size()][]);
    }
    
    public Interval[] mergeOverlappedIntervals(Interval[] intervals) {
        // O(nlgn) sort
        Arrays.sort(intervals, new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                return Integer.compare(o1.start, o2.start);
            }
            
        });
        
        LinkedHashSet<Interval> res = new LinkedHashSet<>();
        
        Interval prev = intervals[0];
        // O(n) for overlaps
        for(int i = 1; i < intervals.length; i++) {
            // overlaps
            if(intervals[i].start < prev.end) {
                // merge intervals
                prev.end = Math.max( prev.end, intervals[i].end);
            }
            // current interval ends
            else {
                res.add(prev);
                prev = intervals[i];
            }
        }
        
        if(!res.contains(prev)) {
            res.add(prev);
        }
        
        Interval[] res1 = new Interval[res.size()];
        
        return res.toArray(res1);
    }
    
    public int[][] insertInterval(int[][] intervals, int[] newInterval) {
        LinkedList<int[]> res = new LinkedList<>();
        
        for(int[] curInterval : intervals) {
            // if new interval completely left to current then take the in the result
            // make the current as new interval as it's position has been taken  
            if(curInterval[0] > newInterval[1]) {
                res.add(newInterval);
                newInterval = curInterval;
            }
            // if new interval is completely right ogf the current then take the current in result
            // new interval remain same with the hope the next interval in input may merge it
            else if(curInterval[1] < newInterval[0]) {
                res.add(curInterval);
            }
            // otherewise merge
            else {
                newInterval = new int[]{Math.min(curInterval[0], newInterval[0]), Math.max(curInterval[1], newInterval[1])};
            }
        }
        
        res.add(newInterval);
        
        return res.toArray(new int[res.size()][]);
    }
    
    public Interval[] insertInterval(Interval[] intervals, Interval newInterval) {
        Set<Interval> res = new LinkedHashSet<>();
        
        for(Interval curInterval : intervals) {
            // if new interval completely left to current then take the in the result
            // make the current as new interval as it's position has been taken  
            if(curInterval.start > newInterval.end) {
                res.add(newInterval);
                newInterval = curInterval;
            }
            // if new interval is completely right of the current then take the current in result
            // new interval remain same with the hope the next interval in input may merge it
            else if(curInterval.end < newInterval.start) {
                res.add(curInterval);
            }
            // otherewise merge
            else {
                newInterval = new Interval(Math.min(curInterval.start, newInterval.start), Math.max(curInterval.end, newInterval.end));
            }
        }
        
        res.add(newInterval);
        
        Interval[] ret = new Interval[res.size()];
        return res.toArray(ret);
    }
    
    public void validIp(List<String> logs) {
        String validIp = "(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])";
        String pattern = validIp+"\\."+validIp+"\\."+validIp+"\\."+validIp;
        Pattern p = Pattern.compile(pattern);
        Set<String> ips = new HashSet<>();
        Matcher m;
        
        for(String log : logs) {
            m = p.matcher(log);
            
            while(m.find()) {
                String ip = m.group().trim();
                if(!ips.contains(ip)) {
                    ips.add(ip);
                    System.out.println(ip);
                }
            }
        }
        
    }
    
    public class DLLListToBSTInplace{
        ListNode head;
        
        private ListNode convert(int start, int end) {
            if(start > end) {
                return null;
            }
            
            int mid = start+(end-start)/2;
            ListNode left = convert(start, mid-1);
            ListNode root = head;
            head = head.next;
            root.prev = left;
            root.next =  convert(mid+1, end);
            
            return root;
        }
        
        public ListNode convert() {
            int n = 0;
            ListNode h = head;
            while(h != null) {
                n++;
                h = h.next;
            }
            
            return convert(0, n-1);
        }
        
        public DLLListToBSTInplace(ListNode head) {
            this.head = head;
        }
    }
    
    public static class BlockingQueue implements Serializable{
        
        private static final long serialVersionUID = 1L;

        // thread safe instance - threads always read the latest updates - read "happens after" write 
        private static volatile BlockingQueue blockingQueueInstance = null;
        
        private Queue<Integer> queue;
        private ReentrantLock lock = new ReentrantLock();
        private Condition notEmptyCondition = lock.newCondition();
        private Condition notFullCondition = lock.newCondition();
        private int capacity;
        
        // making it singleton
        private BlockingQueue(int capacity) {
            if(blockingQueueInstance != null) {
                throw new RuntimeException("SingleTone viloation: use getInstance method to get the instance");
            }
            
            queue = new LinkedList<>();
            this.capacity = capacity; 
        }
        
        // create single instance or return existing
        public static BlockingQueue getInstance(int capacity) {
            // check ti make sure instance not created
            if(blockingQueueInstance == null) {
                // another thread may also try to instantiate at same time
                // so put under a monitor
                synchronized (BlockingQueue.class) {
                    //  double check to make sure
                    if(blockingQueueInstance == null) {
                        blockingQueueInstance = new BlockingQueue(capacity);
                    }
                }
            }
            return blockingQueueInstance;
        }
        
        public void put(int data) throws Exception {
            // take the reentrant lock
            lock.lock();
            lock.lockInterruptibly();
            
            try {
                if(queue.size() == capacity) {
                    notFullCondition.await();
                }
                
                queue.add(data);
                // as we added some we can notify that queue is not empty anymore
                notEmptyCondition.notifyAll();
            }
            finally {
                lock.unlock();
            }
        }
        
        public boolean offer(int data, long timeout, TimeUnit unit) throws Exception  {
            // take the reentrant lock
            lock.lock();
            lock.lockInterruptibly();
            
            try {
                while(queue.size() == capacity) {
                    if(timeout <= 0)
                        return false;
                    notFullCondition.await(timeout, unit);
                }
                
                queue.add(data);
                // as we added some we can notify that queue is not empty anymore
                notEmptyCondition.notifyAll();
            }
            finally {
                lock.unlock();
            }
            
            return false;
        }
        
        // to make it blocking on data availability wait on the producer to fill up
        // use condition to wait till producer notifiies
        public Integer take() throws Exception {
         // take the reentrant lock
            Integer res = null;
            lock.lock();
            lock.lockInterruptibly();
            
            try {
                if(queue.isEmpty()) {
                    notEmptyCondition.await();
                }
                
                res = queue.remove();
                // as we removed some we can notify that queue is not full anymore
                notFullCondition.notifyAll();
            }
            finally {
                lock.unlock();
            }
           
            return res;
        }
        
        public Integer poll(long timeout, TimeUnit unit) throws Exception {
            // take the reentrant lock
            Integer res = null;
            lock.lock();
            lock.lockInterruptibly();
            
            try {
                while(queue.isEmpty()) {
                    if(timeout <= 0)
                        return null;
                    notEmptyCondition.await(timeout, unit);
                }
                
                res = queue.remove();
                // as we removed some we can notify that queue is not full anymore
                notFullCondition.notifyAll();
            }
            finally {
                lock.unlock();
            }
           
            return res;
        }
    }
    
    public ListNode reverse(ListNode head, ListNode reversed) {
        if(head == null) {
            return reversed;
        }
        
        ListNode current = head;
        head = head.next;
        current.next = reversed;
        
        return reverse(head, current);
    }
    
    public ListNode reverseK(ListNode head, ListNode tail, ListNode reversed, int k, int count) {
        if(head == null) {
            return reversed;
        }
        
        if(count == k) {
            tail.next = reverseK(head, null, null, k, 0);
            return reversed;
        }
        else {
            ListNode current = head;
            if(reversed == null) {
                tail = current;
            }
            head = head.next;
            current.next = reversed;
            return reverseK(head, tail, current, k, count+1);
        }
    }
    
    public ListNode merge(ListNode l1, ListNode l2) {
        if(l1 == null || l2 == null) {
            return l1 == null? l2 : l1;
        }
        
        ListNode dummy = new ListNode(0),cur = l1;
        
        while(l1 != null && l2 != null) {
            if(l1.val <= l2.val) {
                cur.next = l1;
                l1 = l1.next;
            }
            else {
                cur.next = l2;
                l2 = l2.next;
            }
        }
        
        if(l1 != null) {
            cur.next = l1;
        }
        else if(l2 != null) {
            cur.next = l2;
        }
        
        return dummy.next;
    }
    
    public ListNode mergeseort(ListNode head) {
        if(head == null) {
            return null;
        }
        
        //cut list into half
        ListNode fast = head, slow = head, left = head, right = null;
        while(fast != null && fast.next != null) {
            if(fast == slow) {
                //cycle detected
                return head;
            }
            
            right = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        right.next = null;
        
        left = mergeseort(left);
        right = mergeseort(right);
        
        //merge
        head = merge(left, right);
        
        return head;
    }
    
    public class maxSumPath{
        int maxSum = Integer.MIN_VALUE;
        
        public int maxSumPath(TreeNode root) {
            maxSumPathDown(root);
            
            return maxSum;
        }
        
        public int maxSumPathDown(TreeNode root) {
            if(root == null) {
                return 0;
            }
            
            //max sum path down (not through root) can be either on left subtree or right subtree
            int leftMaxSumPath = maxSumPathDown(root.left);
            int rightMaxSumPath = maxSumPathDown(root.right);
            
            // on the way compute the sum on the  path through root and update the global max
            // this is because sums from either subtree can be negative 
            // so going through root is not always the max sum path
            maxSum = Math.max(maxSum, leftMaxSumPath+rightMaxSumPath+root.val);
            
            // return the max one ending at root
            return Math.max(leftMaxSumPath, rightMaxSumPath) + root.val;
        }
    }
    
    public class closestLeaf{
        int minDist = Integer.MAX_VALUE;
        TreeNode closest = null;
        
        public int closestLeaf(TreeNode root, int curDist, TreeNode curClosest) {
            if(root == null) {
                minDist = Math.min(minDist, curDist);
                if(minDist == curDist)
                    closest = curClosest;
                
                return curDist;
            }
            
            int left = closestLeaf(root.left, curDist+1, curClosest);
            int right = closestLeaf(root.right, curDist+1, curClosest);
            
            return Math.min(left, right);
        }
    }
    
    //max value less than equal to key
    public int floor(int[] a, int key) {
        int l = 0;
        int h = a.length-1;
        
        if(key < a[l]) {
            return -1;
        }
        
        while(l < h) {
            int mid = l + (h-l)/2;
            
            if(a[mid] == key)
                return mid;
            else if(a[mid] < key) {
                if(mid < a.length-1 && a[mid+1] >= key) {
                    return mid;
                }
                l = mid+1;
            }
            else {
                h = mid-1;
            }
        }
        
        return l;
    }
    
    //min value greater than equal to key
    public int ceil(int[] a, int key) {
        int l = 0;
        int h = a.length-1;
        
        if(key > a[h]) {
            return -1;
        }
        
        while(l < h) {
            int mid = l + (h-l)/2;
            
            if(a[mid] == key)
                return mid;
            else if(a[mid] < key) {
                l = mid + 1;
            }
            else {
                if(mid > 0 && a[mid-1] < key) {
                    return mid;
                }
                h = mid - 1;
            }
        }
        
        return l;
    }
    
    public int threeSumClosest(int[] nums, int target) {
        if(nums.length < 3){
            return -1;
        }
        
        int result = nums[0] + nums[1] + nums[nums.length - 1];
        if(nums.length == 3){
            return result;
        }
        
        Arrays.sort(nums);
        
        for(int i = 0; i < nums.length-2; i++){
            if(i == 0 || (i > 0 && nums[i] != nums[i-1])){
                int j = i+1, k = nums.length-1;
                
                while(j < k){
                    int sum =  nums[i]+nums[j]+nums[k];
                    if(sum == target)
                        return sum;
                    
                    if(sum < target){
                        j++;
                    }
                    else {
                        k--;
                    }
                    
                    result = Math.abs(sum-target) < Math.abs(result-target) ? sum : result;
                }
            }
        }
        
        return result;
    }      
    
    public boolean closer(int k1, int k2, int key) {
        int diff1 = Math.abs(k1 - key);
        int diff2 = Math.abs(k2 - key);
        
        return diff1 <= diff2;
    }
    
    public int minDiffElement(int a[], int key) {
        if(a.length == 0) {
            return -1;
        }
        if(a.length == 1) {
            return 0;
        }
        else if(a.length == 2) {
            return closer(a[0], a[1], key) ? 0 : 1;
        }
        else if(a[0] >= key) {
            return 0;
        }
        else if(a[a.length-1] <= key) {
            return a.length-1;
        }
        
        int l = 0;
        int h = a.length-1;
        int mid = 0;
        
        while(l < h) {
            mid = l + (h-l)/2;
            
            if(a[mid] == key) {
                return mid;
            }
            // mid is already higher; then either mid is closest or it is in a[l..mid-1]
            if(a[mid] > key) {
                if(closer(a[mid], a[mid-1], key)) {
                    return mid;
                }
                h = mid -1;
            }
            // mid is already lesser; then either mid is closest or it is in a[mid+1..h]
            else {
                if(closer(a[mid], a[mid+1], key)) {
                    return mid;
                }
                l = mid+1;
            }
            
        }
        
        
        return -1;
    }
    
    public int minDiffElement(int a[], int l, int h, int key) {
        if(a.length == 0 || l < 0 || h > a.length-1) {
            return -1;
        }
        if(l == h) {
            return l;
        }
        else if(h == l+1) {
            return closer(a[l], a[h], key) ? l : h;
        }
        else if(a[l] >= key) {
            return l;
        }
        else if(a[h] <= key) {
            return h;
        }
        
        while(l < h) {
            int mid = l + (h-l)/2;
            
            if(a[mid] == key) {
                return mid;
            }
            // mid is already higher; then either mid is closest or it is in a[l..mid-1]
            if(a[mid] > key) {
                if(closer(a[mid], a[mid-1], key)) {
                    return mid;
                }
                h = mid -1;
            }
            // mid is already lesser; then either mid is closest or it is in a[mid+1..h]
            else {
                if(closer(a[mid], a[mid+1], key)) {
                    return mid;
                }
                l = mid+1;
            }
            
        }
        
        return -1;
    }
    
//    public List<List<Integer>> threeSum2(int[] a) {
//        List<List<Integer>> res = new ArrayList<>();
//        
//        if(a.length < 3) {
//            return res;
//        }
//        Arrays.sort(a);
//        
//        int l = 0;
//        int h = a.length-1;
//        while(l < h && (h-l+1) >=3){
//            int third = 0 - (a[l]+a[h]);
//            int closestKey = minDiffElement(a, l+1, h-1, third);
//            int sum = a[l]+a[h]+a[closestKey];
//            
//            if(sum == 0){
//                res.add(Arrays.asList(a[l], a[closestKey], a[h]));
//                l++;
//                h--;
//                // handle duplicate cases
//                while(l < h && (a[l] == a[l-1]) && (a[h] == a[h+1])) {
//                    l++;
//                    h--;
//                }
//            }
//            else if(sum < 0){
//                l++;
//            }
//            else{
//                h--;
//            }
//        }
//        
//        return res;
//    }
    
    // accepted 14 ms
    // O(nlgn) < Oder < O(n^2)
    public List<List<Integer>> threeSum3(int[] nums) {
        List<List<Integer>> result = new LinkedList<>();
        Arrays.sort(nums);
        
        for(int i = 0; i < nums.length-2; i++){
            if(i == 0 || (i > 0 && nums[i] != nums[i-1])){
                // unoptiomized 
                int j = i+1, k = nums.length-1;
                // Each time we increment i, we pick the next k by selecting the last number in the 
                // list, which is also the maximum number. Now, once i gets closer and closer to zero (in the sorted list), 
                // this k will be less appropriate and we will end up doing a linear search from k 
                // towards i to get a smaller positive number that will give us a zero sum. 
                // The optimal k can be found more efficiently with a binary search, which makes 
                // the overall algorithm more efficient again.
                // A similar argument applies to j. If we consider the maximum number, it may make less sense 
                // to make j become i+1 for its first candidate. Namely if i+j+k==0, then j should be -i + -k.
                j = binarySearchClosest(nums, i + 1, nums.length - 2, -(nums[i] + nums[nums.length-1]));
                k = binarySearchClosest(nums, j + 1, nums.length - 1, -(nums[i] + nums[j]));
                
                while(j < k && k < nums.length){
                    int sum =  nums[i]+nums[j]+nums[k];
                    
                    if(sum < 0){
                        j++;
                    }
                    else if(sum > 0){
                        k--;
                    }
                    else{         
                        result.add(Arrays.asList(nums[i], nums[j], nums[k]));
                        while(j < k && nums[j] == nums[j+1]) j++;
                        while(j < k && nums[k] == nums[k-1]) k--;
                        j++;
                        k--;
                    }
                }
            }
        }
        
        return result;
    }   
    
    public List<List<Integer>> kSum(int[] nums, int targetSum, int k) {
        if(nums == null || nums.length < k || k < 2) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        return kSumHelperOnSorted(nums, targetSum, k, 0);
    }
    
    public List<List<Integer>> kSumHelperOnSorted(int[] nums, int targetSum, int k, int start) {
        int n = nums.length;
        ArrayList<List<Integer>> result = new ArrayList<List<Integer>>();
        
        if(start >= n || n-start+1 < k || k < 2 || targetSum < nums[start]*k || targetSum > nums[n-1]*k) {
            return result;
        }
        
        if(k == 2) {
            return twoSumOnSorted(nums, start, n-1, targetSum);
        }
        else {
            
            for(int i = start; i < n - k + 1; i++) {
                // skip duplicates
                if(i == start || (i > start && nums[i-1] != nums[i])){
                    // for each nums[i] recursively find the next k-1 numbers that along with nums[i] can make the total sum to be targetSum
                    List<List<Integer>> subProblemResult = kSumHelperOnSorted(nums, targetSum-nums[i], k-1, i+1);
                    
                    // add nums[i] to the (k-1) subproblem result list to make it k sum result
                    for(List<Integer> res : subProblemResult) {
                        res.add(0, nums[i]);
                    }
                    
                    result.addAll(subProblemResult);
                }
            }
        }
        
        return result;
    }
    
    public List<List<Integer>> twoSumOnSorted(int[] nums, int i, int j, int targetSum){
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        
        while(i < j) {
            int sum = nums[i]+nums[j];
            if(sum == targetSum) {
                List<Integer> res = new ArrayList<>(Arrays.asList(nums[i], nums[j]));
                result.add(res); 
                i++;
                j--;
                // skip dupicates
                while(i < j && nums[i] == nums[i-1]) i++;
                while(i < j && nums[j] == nums[j+1]) j--;
            }
            else if(sum < targetSum) {
                i++;
            }
            else {
                j--;
            }
        }
        
        return result;
    }
    
    public int binarySearchClosest(int a[], int l, int h, int key) {
        if(a.length == 0 || l < 0 || h > a.length-1) {
            return -1;
        }
        
        while(true) {
            if(h < l){
                return l;
            }
            int mid = l + (h-l)/2;
            
            if(a[mid] == key) {
                return mid;
            }
            if(a[mid] > key) {
                h = mid -1;
            }
            else {
                l = mid+1;
            }
        }
    }
    
    public class PartitionLabels {
        public class Interval{
            public int start;
            public int end;
            
            public Interval(int start, int end){
                this.start = start;
                this.end = end;
            }
        }
        public List<Integer> partitionLabels(String S) {
            if(S == null || S.length() == 0){
                return null;
            }
            // one interval per character
            // insert order is based on first appearance of the character
            LinkedHashMap<Integer, Interval> map = new LinkedHashMap<Integer, Interval>(26);
            // build the intervalMap
            for(int i = 0; i < S.length(); i++){
                int c = S.charAt(i) - 'a';
                if(!map.containsKey(c)){
                    map.put(c, new Interval(i, i));
                }
                else{
                    map.get(c).end = i;
                }
            }
            
            LinkedHashSet<Interval> res = new LinkedHashSet<>();
            List<Integer> result = new ArrayList<>();
            Iterator it = map.entrySet().iterator();
            Map.Entry<Integer, Interval> entry = (Map.Entry<Integer, Interval>) it.next();
            Interval prev = entry.getValue();
            
            // now start merging intervals
            while(it.hasNext()){
                entry = (Map.Entry<Integer, Interval>) it.next();
                Interval cur = entry.getValue();
                
                if(cur.start < prev.end){
                    prev = new Interval(prev.start, Math.max(prev.end, cur.end));
                }
                else{
                    res.add(prev);
                    result.add(prev.end - prev.start + 1);
                    prev = cur;
                }
            }
            
            if(!res.contains(prev)){
                res.add(prev);
                result.add(prev.end - prev.start + 1);
            }
            
            return result;
        }
    }
    
    public int fourListSum(int A[], int B[], int C[], int D[]) {
        Map<Integer, Integer> abCount = new HashMap<>();
        Arrays.stream(A).forEach(a -> {
            Arrays.stream(B).forEach(b -> {
                abCount.put(a+b, 1+abCount.getOrDefault(a+b,0));
            });
        });
        
        AtomicInteger counter = new AtomicInteger(0);
        Arrays.stream(C).forEach(c -> {
            Arrays.stream(D).forEach(d -> {
                counter.addAndGet(abCount.getOrDefault(-(c+d), 0));
            });
        });
        
        return counter.get();
    }
    
    /**
    1. Find the largest index k such that nums[k] < nums[k + 1]. 
    2. If no such index exists, the permutation is sorted in descending order, just reverse it to ascending order and        we are done. For example, the next permutation of [3, 2, 1] is [1, 2, 3].
    3. Find the largest index l greater than k such that nums[k] < nums[l].
    4 Swap the value of nums[k] with that of nums[l].
    5. Reverse the sequence from nums[k + 1] up to and including the fi
    */
    public void nextPermutation(int[] nums) {
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
    
    public void reverse(int A[], int i, int j){
        while(i < j){
            swap(A, i++, j--);
        }
    }
    
    public void swap(final int[] a, final int i, final int j) {
        if (i == j || i < 0 || j < 0 || i > a.length - 1 || j > a.length - 1) {
            return;
        }
        a[i] ^= a[j];
        a[j] ^= a[i];
        a[i] ^= a[j];
    }
    
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Set<Integer> visited = new HashSet<>();
        permutation(nums, new LinkedList<>(), res, visited);
        return res;
    }
    
    public void permutation(int[] nums, LinkedList<Integer> cur, List<List<Integer>> res, Set<Integer> visited){
        if(cur.size() == nums.length){
            res.add(new LinkedList<>(cur));
        }
        else{
            for(int i = 0; i < nums.length; i++){
                if(visited.contains(nums[i])){
                    continue;
                }
                visited.add(nums[i]);
                cur.add(nums[i]);
                permutation(nums, cur, res, visited);
                cur.removeLast();
                visited.remove(nums[i]);
            }
        }
    }
    
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Set<Integer> visited = new HashSet<>();
        Arrays.sort(nums);
        permutationUnique(nums, new LinkedList<>(), res, visited);
        return res;
    }
    
    public void permutationUnique(int[] nums, LinkedList<Integer> cur, List<List<Integer>> res, Set<Integer> visited){
        if(cur.size() == nums.length){
            res.add(new LinkedList<>(cur));
        }
        else{
            for(int i = 0; i < nums.length; i++){
                if(visited.contains(i) || (i > 0 && nums[i] == nums[i-1] && !visited.contains(i-1))){
                    continue;
                }
                visited.add(i);
                cur.add(nums[i]);
                permutationUnique(nums, cur, res, visited);
                cur.removeLast();
                visited.remove(i);
            }
        }
    }
    
    public static void main(String[] args) {
        
        Test t = new Test();
        
        int ft = t.fourListSum(new int[] {-1,-1}, new int[] {-1,1}, new int[] {-1,1}, new int[] {1,-1});
        
        List<List<Integer>> rest = t.kSum(new int[] {1, 0, -1, 0, -2, 2}, 0, 4);
        
        int re = t.minDiffElement(new int[] {1, 3, 6, 7, 9, 10}, -1);
        
        int re1 = t.binarySearchClosest(new int[] {1, 3, 6, 7, 9, 10}, 0, 5, -1);
        
        PartitionLabels s = t.new PartitionLabels();
        @SuppressWarnings("unused")
        List<Integer> parts = s.partitionLabels("ababcbacadefegdehijhklij");
        
        int floor = t.floor(new int[] {-1,0,3,3,5,6,8}, 4);
        int ceil = t.ceil(new int[] {-1,0,3,3,5,6,8}, 4);
        
        ListNode dummy = null;
        ListNode head = t.new ListNode(1);
        dummy = head;
        head.next = t.new ListNode(2);
        head = head.next;
        head.next = t.new ListNode(3);
        head = head.next;
        head.next = t.new ListNode(4);
        head = head.next;
        head.next = t.new ListNode(5);
        head = head.next;
        head.next = t.new ListNode(6);
        head = head.next;
        head.next = t.new ListNode(7);
        head = head.next;
        head.next = t.new ListNode(8);
        
        //ListNode head2 = t.new DLLListToBSTInplace(dummy).convert();
        
        head = t.reverseK(dummy, null, null, 3, 0);
        
        /**
         *      4
         *     / \
         *    2   6
         *   / \ / \
         *   1 3 5  7
         *           \
         *            8
         *    
         *     
         */
        
        head = t.oddEvenList(dummy);
        System.out.println();
        
        TreeNode root6 = t.new TreeNode(6);
        TreeNode root4 = t.new TreeNode(4);
        TreeNode root8 = t.new TreeNode(8);
        TreeNode root1 = t.new TreeNode(1);
        TreeNode root5 = t.new TreeNode(5);
        TreeNode root11 = t.new TreeNode(11);
        TreeNode root10 = t.new TreeNode(10);
        
        root6.left = root4;
        root6.right = root8;
        root4.left = root1;
        root4.right = root5;
        root8.right = root11;
        root11.left = root10;
        
        //List<Integer> llist = t.preorderTraversal(root6);
        
        //TreeNode dll = t.inorderDLListInplace(root6);
        //TreeNode dll = t.inorderDLListInplaceRecursive(root6);
        TreeNode dll = t.inorderCircularDLListInplace(root6);
        
        TreeNode tail = dll;
        while(dll != null && dll != tail) {
            System.out.print(dll.val+" ");
            tail = dll;
            dll = dll.right;
        }
        
        System.out.println();
        while(tail != null) {
            System.out.print(tail.val+" ");
            tail = tail.left;
        }
        
        
        t.validIp(Arrays.asList(new String[] {"ewdjbwouhfsu255.248.89.9sdssdadsa0.0.0.0.sdbkjdb1.34.46.7wdfdsfsd23.34.56.sfdfsdfs00.0.0.0.0.dfsfs"}));
        
        Interval[] ints = new Interval[] {
                t.new Interval(1,11,5), 
                t.new Interval(2,6,7), 
                t.new Interval(3,13,9), 
                t.new Interval(12,7,16), 
                t.new Interval(14,3,25), 
                t.new Interval(19,18,22), 
                t.new Interval(23,13,29), 
                t.new Interval(24,4,28) };
        
        Interval[] ov = t.mergeOverlappedIntervals(ints);
        
        List<List<Integer>> resultt = new ArrayList<>();
        System.out.println(resultt);
        List<Integer> curr = new ArrayList<>();
        
        int[] res = t.searchRange(new int[] {5,7,7,8,8,10}, 8);
        System.out.println(res);
        
        String minl = t.minLenSuperSubString("adobecodebanc", "abc");
        
        int m[][] = new int[][] {{9, 9, 4}, {6, 6, 8}, {2, 1, 0}};
        List<Integer> ers = t.walkDFS(m);
        
        Graph g = t.new Graph();
        g.vertices = new int[] {2, 3, 4, 6, 5, 7, 4, 1, 8, 3};
        g.edges = new Edge[g.vertices.length+1][g.vertices.length+1];
        
        g.edges[0][1] = g.new Edge(0, 1, 3);
        g.edges[0][2] = g.new Edge(0, 2, 4);
        g.edges[1][3] = g.new Edge(1, 3, 6);
        g.edges[1][4] = g.new Edge(1, 4, 5);
        g.edges[2][4] = g.new Edge(2, 4, 5);
        g.edges[2][5] = g.new Edge(2, 5, 7);
        g.edges[3][6] = g.new Edge(3, 6, 4);
        g.edges[3][7] = g.new Edge(3, 7, 1);
        g.edges[4][7] = g.new Edge(4, 7, 1);
        g.edges[4][8] = g.new Edge(4, 8, 8);
        g.edges[5][8] = g.new Edge(5, 8, 8);
        g.edges[5][9] = g.new Edge(5, 9, 3);
        
        List<Integer> spath = g.shortestPath(0, 9);
               

        List<List<String>> input = Arrays.asList(Arrays.asList(new String[] {"a", "b", "c"}), Arrays.asList(new String[] {"d", "e"}), Arrays.asList(new String[] {"f"}));
        List<List<String>> result = new ArrayList<>();
        String[] cur = new String[input.size()];
        
        t.permList(input, cur, 0, result);
    }
}
