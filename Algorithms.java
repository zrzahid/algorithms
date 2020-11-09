package test;

import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeSet;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import test.Test.Graph.Edge;
import test.Test.GraphTraversals.Trie;
import test.Test.IntervalOps.PartitionLabels;

public class Test {

    public void reverse(int A[], int i, int j) {
        while (i < j) {
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

    public void shiftRight(int[] a, int s, int e) {
        int temp = a[e];
        for (int i = e; i > s; i--) {
            a[i] = a[i - 1];
        }
        a[s] = temp;
    }

    public class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode next;
        int size;
        int height;

        TreeNode(int x) {
            val = x;
        }

    }

    public class ListNode {

        int val;
        ListNode next;
        ListNode prev;

        public ListNode(int x) {
            val = x;
        }

        public ListNode(int x, ListNode nxt) {
            val = x;
            next = nxt;
        }
    }

    public class Interval {
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

    public class Graph {

        class Edge {
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
    }

    public class Building {

        int l;
        int h;
        int r;

        public Building(int left, int height, int right) {
            l = left;
            h = height;
            r = right;
        }
    }

    public class Strip {

        int l;
        int h;

        public Strip(int left, int height) {
            l = left;
            h = height;
        }

        @Override
        public String toString() {
            return "(" + l + ", " + h + ")";
        }
    }

    public interface JSONTokenStream {
        boolean hasNext();

        JSONToken next();
    }

    public interface JSONToken {
        int type(); // 0=StartObject, 1=EndObject, 2=Field

        String name(); // null for EndObject

        String value(); // null for StartObject, EndObject
    }

    public class JsonNode {

        int type;
        String name;
        String value;
        TreeSet<JsonNode> child;

        public JsonNode(JSONToken token) {
            name = token.name();
            value = token.value();
            type = token.type();
            child = new TreeSet<JsonNode>();
        }

        public void addChild(JSONToken tok) {
            JsonNode child = new JsonNode(tok);
            this.child.add(child);
        }

        public boolean equals(JsonNode n1, JsonNode n2) {
            return (n1.type == n2.type) && (n1.name.equals(n2.name)) && (n1.value == n2.value)
                    && (n1.child.size() == n2.child.size()) && (n1.child.equals(n2.child));
        }

    }

    public class MatrixElement implements Comparable<MatrixElement> {
        public int val;
        public int row;
        public int col;

        public MatrixElement(int val, int row, int col) {
            this.val = val;
            this.row = row;
            this.col = col;
        }

        @Override
        public int compareTo(MatrixElement o) {
            return Integer.compare(this.val, o.val);
        }
    }

    public class Job implements Comparable<Job> {
        public int start;
        public int finish;
        public int weight;
        public int mode = 0;

        public Job(int start, int finish) {
            this.start = start;
            this.finish = finish;
            this.weight = 1;
        }

        public Job(int start, int finish, int weight) {
            this.start = start;
            this.finish = finish;
            this.weight = weight;
        }

        @Override
        public int compareTo(Job o) {
            if (mode == 1) {
                return Integer.compare(this.finish, o.start);
            } else {
                return Integer.compare(this.finish, o.finish);
            }
        }

        @Override
        public String toString() {
            return "[" + start + "," + finish + "," + weight + "]";
        }

        public void print() {
            System.out.println(this.toString());
        }
    }

    public class Point implements Comparable<Point> {

        public double x;
        public double y;

        public Point(final double x, final double y) {
            this.x = x;
            this.y = y;
        }

        public double getDist() {
            return x * x + y * y;
        }

        @Override
        public int compareTo(Point o) {
            int c = Double.compare(getDist(), o.getDist());
            if (c == 0) {
                c = Double.compare(x, o.x);
                if (c == 0) {
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

    class TreeTraversals {

        public List<Integer> inorderTraversal(TreeNode root) {
            Stack<TreeNode> stack = new Stack<>();
            List<Integer> res = new LinkedList<>();

            while (root != null || !stack.isEmpty()) {
                // go far lett (min node) and stack up all nodes
                while (root != null) {
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

        public List<Integer> preorderTraversal(TreeNode root) {
            List<Integer> result = new ArrayList<>();
            Deque<TreeNode> stack = new ArrayDeque<>();
            TreeNode p = root;
            while (!stack.isEmpty() || p != null) {
                if (p != null) {
                    stack.push(p);
                    result.add(p.val); // Add before going to children
                    p = p.left;
                } else {
                    TreeNode node = stack.pop();
                    p = node.right;
                }
            }
            return result;
        }

        /**
         * 
         * 3 / \ 9 20 / \ 15 7
         * 
         * @param root
         * 
         *             preorder: 3, 9, 20, 15, 7
         */
        public void preorder(TreeNode root, boolean reverse) {
            if (root == null) {
                return;
            }

            System.out.println(root.val);
            if (reverse) {
                preorder(root.right, !reverse);
                preorder(root.left, !reverse);
            } else {
                preorder(root.left, !reverse);
                preorder(root.right, !reverse);
            }
        }
        
        /**
         * We will basically traverse the tree in pre-order (root -> left -> right) such a way 
         * that we assign one higher priority when we go left and one lower priority when we go right. 
         * Then we will basically put all the nodes with same priority value in a map. 
         * 
         * Once the traversal is done we can print the map nodes in the order of priority, same priority 
         * nodes in the same line. For example, the following tree shows the assigned priority for each 
         * node in vertical traverse order [lower value means higher priority]. 
         * 
         *                   1,0
         *                  /   \
         *                2,-1    3,1
         *              /  \      /  \
         *            4,-2   5,0 6,0  7,2
         *           
         *           map :
         *           -2 -> {4}
         *           -1 -> {2}
         *            0 -> {1, 5, 6}
         *            1 -> {3}
         *            2 -> {7}
         * 
         * @param root
         */
        public void verticalTrversal(TreeNode root){
            int[] minmax = new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE};
            Map<Integer, ArrayList<TreeNode>> verticals = new HashMap<Integer, ArrayList<TreeNode>>();
            traverse(root, verticals, 0, minmax);
            for(int i = minmax[0]; i<=minmax[1]; i++){
                if(verticals.containsKey(i)){
                    for(TreeNode vnode : verticals.get(i)){
                        System.out.print(vnode.val+",");
                    }
                    System.out.println();
                }
            }
            
        }

        private void traverse(TreeNode node, Map<Integer, ArrayList<TreeNode>> verticals, int score, int[] minmax){
            if(!verticals.containsKey(score)){
                verticals.put(score, new ArrayList<TreeNode>());
            }
            
            verticals.get(score).add(node);
            minmax[0] = Math.min(minmax[0], score);
            minmax[1] = Math.max(minmax[1], score);
            
            if(node.left != null){
                traverse(node.left, verticals, score-1, minmax);
            }
            if(node.right != null){
                traverse(node.right, verticals, score+1, minmax);
            }
        }

        public List<Integer> postorderTraversal(TreeNode root) {
            LinkedList<Integer> result = new LinkedList<>();
            Deque<TreeNode> stack = new ArrayDeque<>();
            TreeNode p = root;
            while (!stack.isEmpty() || p != null) {
                if (p != null) {
                    stack.push(p);
                    result.addFirst(p.val); // Reverse the process of preorder
                    p = p.right; // Reverse the process of preorder
                } else {
                    TreeNode node = stack.pop();
                    p = node.left; // Reverse the process of preorder
                }
            }
            return result;
        }

        public List<List<Integer>> levelOrder(TreeNode root) {
            List<List<Integer>> res = new ArrayList<>();
            
            if(root == null){
                return res;
            }
            List<Integer> cur = new ArrayList<>();
            
            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);
            int count = 1;
            while (!queue.isEmpty()) {
                TreeNode node = queue.poll();
                count--;

                cur.add(node.val);
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }

                if (count == 0) {
                    res.add(cur);
                    cur = new ArrayList<>();
                    count = queue.size();
                }
            }
            
            return res;
        }
        
        public void connectLevelOrder(TreeNode root) {
            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);

            TreeNode node = null;
            int count = 1;
            while (!queue.isEmpty()) {
                if (node == null) {
                    node = queue.poll();
                    node.next = null;
                } else {
                    node.next = queue.poll();
                    node = node.next;
                }
                count--;

                System.out.print(node.val + " ");
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }

                if (count == 0) {
                    node = null;
                    System.out.println("");
                    count = queue.size();
                }
            }
        }

        public List<Integer> rightSideViewOfBT(TreeNode root) {
            List<Integer> res = new LinkedList<>();

            if (root == null) {
                return res;
            }

            Queue<TreeNode> queue = new LinkedList<>();
            queue.add(root);
            int perLevelNodeCount = 1;

            while (!queue.isEmpty()) {
                TreeNode node = queue.remove();
                perLevelNodeCount--;

                // now for BFS walk, add the childrens to queue
                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }

                // if count becomes zero that means this is the right most node in the level
                // add it to result
                // also reset count for next level based on queue size
                if (perLevelNodeCount == 0) {
                    res.add(node.val);
                    perLevelNodeCount = queue.size();
                }
            }

            return res;
        }

        public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
            List<List<Integer>> result = new LinkedList<>();

            if (root == null) {
                return result;
            }

            Queue<TreeNode> queue = new LinkedList<TreeNode>();
            queue.add(root);
            int count = 1;
            int level = 0;

            LinkedList<Integer> curLevel = new LinkedList<>();
            while (!queue.isEmpty()) {
                TreeNode cur = queue.remove();
                count--;

                // add to cur level list
                if (level % 2 == 0)
                    curLevel.addLast(cur.val);
                else
                    curLevel.addFirst(cur.val);

                if (cur.left != null) {
                    queue.add(cur.left);
                }
                if (cur.right != null) {
                    queue.add(cur.right);
                }

                // end of current level
                if (count == 0) {
                    level++;
                    count = queue.size();
                    result.add(curLevel);
                    curLevel = new LinkedList<>();
                }
            }

            return result;
        }

        public TreeNode rightMostCousin(TreeNode root, int targetKey) {
            LinkedList<TreeNode> q = new LinkedList<TreeNode>();

            int count = 0;
            q.add(root);
            count++;
            boolean targetLevel = false;

            while (!q.isEmpty()) {
                TreeNode node = q.remove();
                count--;
                if ((node.left != null && node.left.val == targetKey)
                        || (node.right != null && node.right.val == targetKey))
                    targetLevel = true;

                if (node.left != null)
                    q.add(node.left);
                if (node.right != null)
                    q.add(node.right);

                if (count == 0) {
                    count = q.size();
                    if (targetLevel) {
                        TreeNode cousin = null;
                        while (!q.isEmpty()) {
                            cousin = q.remove();
                        }

                        return cousin;
                    }
                }
            }

            return null;
        }

        public TreeNode rightMostCousin2(TreeNode root, int targetKey) {
            LinkedList<TreeNode> q = new LinkedList<TreeNode>();

            int count = 0;
            q.add(root);
            count++;
            boolean targetLevel = false;

            while (!q.isEmpty()) {
                TreeNode node = q.remove();
                count--;
                if (node.val == targetKey)
                    targetLevel = true;

                if (node.left != null)
                    q.add(node.left);
                if (node.right != null)
                    q.add(node.right);

                if (count == 0) {
                    count = q.size();
                    if (targetLevel) {
                        if (node.val != targetKey)
                            return node;
                        else
                            return null;
                    }
                }
            }

            return null;
        }
        
        public void inorderMorris(TreeNode root) {

            TreeNode cur = root;
            TreeNode pre = null;

            while (cur != null) {
                // if no left child then print current and go to right subtree
                if (cur.left == null) {
                    System.out.print(cur.val + " ");
                    cur = cur.right;
                } else {
                    // left subtree exists. Inorder traversal has to come back to this current node
                    // after the left subtree is traversed
                    // usually we can achieve it by pushing the cur node to stack.
                    // but without any stack how do we achieve it?
                    // we know that the last traversed node in the left subtree will be
                    // the left subtree, which is the predecessor for current node.
                    // So, we can make the traverse come back to current node naturally
                    // by threading the right most child's right pointer to the current node.

                    // so first find the predecessor of cur and thread it's right pointer to current
                    // If the threaded pointer was set already then we have naturally
                    // came back to the root through the threaded pointer. So, not need to thread
                    // agains.
                    pre = cur.left;
                    while (pre.right != null && pre.right != cur) {
                        pre = pre.right;
                    }

                    // if the predecessor is found than thread it's right pointer to current
                    // and then traverse to next left
                    if (pre.right == null) {
                        pre.right = cur;
                        cur = cur.left;
                    }
                    // if threaded pointer was already set in previous step than we have naturally
                    // came back to the root through the threaded pointer.
                    // so, print the root, unthead the predecessor, and go to right
                    else {
                        System.out.print(cur.val + " ");
                        pre.right = null;
                        cur = cur.right;
                    }
                }
            }
        }

        public void preMorris(TreeNode root) {

            TreeNode cur = root;
            TreeNode pre = null;

            while (cur != null) {
                // if no left child then print current and go to right subtree
                if (cur.left == null) {
                    System.out.print(cur.val + " ");
                    cur = cur.right;
                } else {
                    // left subtree exists. Inorder traversal has to come back to this current node
                    // after the left subtree is traversed
                    // usually we can achieve it by pushing the cur node to stack.
                    // but without any stack how do we achieve it?
                    // we know that the last traversed node in the left subtree will be
                    // the left subtree, which is the predecessor for current node.
                    // So, we can make the traverse come back to current node naturally
                    // by threading the right most child's right pointer to the current node.

                    // so first find the predecessor of cur and thread it's right pointer to current
                    // If the threaded pointer was set already then we have naturally
                    // came back to the root through the threaded pointer. So, not need to thread
                    // agains.
                    pre = cur.left;
                    while (pre.right != null && pre.right != cur) {
                        pre = pre.right;
                    }

                    // if the predecessor is found than thread it's right pointer to current
                    // and then traverse to next left
                    if (pre.right == null) {
                        pre.right = cur;
                        System.out.print(cur.val + " "); // preorder - print as soon as we traversea node
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
    }

    class BalancedBinaryTree {

        public int size(final TreeNode node) {
            return node == null ? 0 : node.size;
        }

        public int height(final TreeNode node) {
            return node == null ? 0 : node.height;
        }

        public TreeNode rotateLeft(final TreeNode root) {
            final TreeNode newRoot = root.right;
            final TreeNode leftSubTree = newRoot.left;

            newRoot.left = root;
            root.right = leftSubTree;

            root.height = Math.max(height(root.left), height(root.right)) + 1;
            newRoot.height = Math.max(height(newRoot.left), height(newRoot.right)) + 1;

            newRoot.size = size(newRoot.left) + size(newRoot.right) + 1;
            newRoot.size = size(newRoot.left) + size(newRoot.right) + 1;

            return newRoot;
        }

        public TreeNode rotateRight(final TreeNode root) {
            final TreeNode newRoot = root.left;
            final TreeNode rightSubTree = newRoot.right;

            newRoot.right = root;
            root.left = rightSubTree;

            root.height = Math.max(height(root.left), height(root.right)) + 1;
            newRoot.height = Math.max(height(newRoot.left), height(newRoot.right)) + 1;

            newRoot.size = size(newRoot.left) + size(newRoot.right) + 1;
            newRoot.size = size(newRoot.left) + size(newRoot.right) + 1;

            return newRoot;
        }

        public TreeNode insertIntoAVL(final TreeNode node, final int key, final int count[], final int index) {
            if (node == null) {
                return new TreeNode(key);
            }

            if (node.val > key) {
                node.left = insertIntoAVL(node.left, key, count, index);
            } else {
                node.right = insertIntoAVL(node.right, key, count, index);

                // update smaller elements count
                count[index] = count[index] + size(node.left) + 1;
            }

            // update the size and height
            node.height = Math.max(height(node.left), height(node.right)) + 1;
            node.size = size(node.left) + size(node.right) + 1;

            // balance the tree
            final int balance = height(node.left) - height(node.right);
            // left-left
            if (balance > 1 && node.val > key) {
                return rotateRight(node);
            }
            // right-right
            if (balance < -1 && node.val > key) {
                return rotateLeft(node);
            }
            // left-right
            if (balance > 1 && node.val < key) {
                node.left = rotateLeft(node.left);
                return rotateRight(node);
            }
            // right-left
            if (balance > 1 && node.val < key) {
                node.right = rotateRight(node.right);
                return rotateLeft(node);
            }

            return node;
        }

        public int[] smallerCountOnRightSlow(final int[] X) {
            int[] smaller = new int[X.length];
            for (int i = 0; i < X.length; i++) {
                for (int j = i + 1; j < X.length; j++) {
                    if (X[j] <= X[i]) {
                        smaller[i]++;
                    }
                }
            }

            return smaller;
        }

        // faster using balanced BT
        public int[] countSmallerOnRight(final int[] in) {
            final int[] smaller = new int[in.length];

            TreeNode root = null;
            for (int i = in.length - 1; i >= 0; i--) {
                root = insertIntoAVL(root, in[i], smaller, i);
            }

            return smaller;
        }
    }

    class BinaryTree {

        public void allPaths(TreeNode root, String path, ArrayList<String> paths) {
            if (root == null) {
                return;
            }

            path = path + (path.isEmpty() ? "" : "-->") + root.val;

            if (root.left == null && root.right == null) {
                System.out.println("path > " + path);
                paths.add(path);
                return;
            }

            allPaths(root.left, path, paths);
            allPaths(root.right, path, paths);
        }

        public TreeNode LCA(TreeNode root, TreeNode x, TreeNode y) {
                if (root == null)
                    return null;
                if (root == x || root == y)
                    return root;
    
                TreeNode leftSubTree = LCA(root.left, x, y);
                TreeNode rightSubTree = LCA(root.right, x, y);
    
                // x is in one subtree and and y is on other subtree of root
                if (leftSubTree != null && rightSubTree != null)
                    return root;
                // either x or y is present in one of the subtrees of root or none present in
                // either side of the root
                return leftSubTree != null ? leftSubTree : rightSubTree;
        }

        public int getLevel(TreeNode root, int count, TreeNode node) {
            if (root == null) {
                return 0;
            }

            if (root == node) {
                return count;
            }

            int leftLevel = getLevel(root.left, count + 1, node);
            if (leftLevel != 0) {
                return leftLevel;
            }
            int rightLevel = getLevel(root.right, count + 1, node);
            return rightLevel;
        }

        public int shortestDistance(TreeNode root, TreeNode a, TreeNode b) {
            if (root == null) {
                return 0;
            }

            TreeNode lca = LCA(root, a, b);
            // d(a,b) = d(root,a) + d(root, b) - 2*d(root, lca)
            return getLevel(root, 1, a) + getLevel(root, 1, b) - 2 * getLevel(root, 1, lca);
        }

        public int maxDepth(TreeNode root) {

            if (root == null) {
                return 0;
            }

            int leftDepth = maxDepth(root.left);
            int rightDepth = maxDepth(root.right);

            return Math.max(leftDepth, rightDepth) + 1;
        }

        // diameter using depth
        public int diameter(TreeNode root) {
            // D(T) = max{D(T.left), D(T.right), LongestPathThrough(T.root)}
            if (root == null) {
                return 0;
            }

            int leftHeight = maxDepth(root.left);
            int rightHeight = maxDepth(root.right);

            int leftDiameter = diameter(root.left);
            int rightDiameter = diameter(root.right);

            return Math.max(Math.max(leftDiameter, rightDiameter), leftHeight + rightHeight + 1);
        }

        // compute directly diameter and height together
        public int diameter(TreeNode root, int[] height) {
            if (root == null) {
                height[0] = 0;
                return 0;
            }

            int[] leftHeight = { 0 }, rightHeight = { 0 };
            int leftDiam = diameter(root.left, leftHeight);
            int rightDiam = diameter(root.right, rightHeight);

            height[0] = Math.max(leftHeight[0], rightHeight[0]) + 1;

            return Math.max(Math.max(leftDiam, rightDiam), leftHeight[0] + rightHeight[0] + 1);
        }

        public void mirrorTree(TreeNode root) {
            if (root == null) {
                return;
            }

            mirrorTree(root.left);
            mirrorTree(root.right);

            TreeNode temp = root.right;
            root.right = root.left;
            root.left = temp;
        }

        public Node connect(Node root) {
            // visit each level and connect it;s childrens
            if (root == null) {
                return null;
            }

            Node cur = null;
            Node preRoot = root;
            while (preRoot.left != null) {

                cur = preRoot;

                while (cur != null) {
                    cur.left.next = cur.right;
                    if (cur.next != null) {
                        cur.right.next = cur.next.left;
                    }
                    cur = cur.next;
                }

                preRoot = preRoot.left;
            }

            return root;
        }

        // DLL
        public TreeNode inorderDLListInplace(TreeNode root) {
            Stack<TreeNode> stack = new Stack<>();
            TreeNode head = null;
            TreeNode curHead = null;

            while (root != null || !stack.isEmpty()) {
                // go far lett (min node) and stack up all nodes
                while (root != null) {
                    stack.push(root);
                    root = root.left;
                }

                // now pop the min node
                root = stack.pop();
                // use right pointer for link
                if (head == null) {
                    head = root;
                    curHead = head;
                } else {
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

            if (node1 == null || node2 == null) {
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

            if (root == null) {
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

        private int[] preorder;
        private int[] inorder;

        public TreeNode buildTree(int[] preorder, int[] inorder) {
            this.preorder = preorder;
            this.inorder = inorder;
            // Preorder traversing implies that PRE[0] is the root node.
            // Then we can find this PRE[0] in IN, say it's IN[5].
            // Now we know that IN[5] is root, so we know that IN[0] - IN[4] is on the left side, IN[6] to the end is on the right side.
            // Recursively doing this on subarrays, we can build a tree out of it :)
            return buildTreeHelper(0, inorder.length - 1, 0);
        }

        public int findRootIndex(int istart, int iend, int nextPre) {
            for (int i = istart; i <= iend; i++) {
                if (this.preorder[nextPre] == this.inorder[i]) {
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

        public class closestLeaf {
            int minDist = Integer.MAX_VALUE;
            TreeNode closest = null;

            public int closestLeaf(TreeNode root, int curDist, TreeNode curClosest) {
                if (root == null) {
                    minDist = Math.min(minDist, curDist);
                    if (minDist == curDist)
                        closest = curClosest;

                    return curDist;
                }

                int left = closestLeaf(root.left, curDist + 1, curClosest);
                int right = closestLeaf(root.right, curDist + 1, curClosest);

                return Math.min(left, right);
            }
        }

        public int findCLosestBST(TreeNode node, int key, int minDiff, int bestResult) {
            int diff = Math.abs(node.val - key);

            if (diff < minDiff) {
                minDiff = diff;
                bestResult = node.val;
            }

            if (minDiff == 0) {
                return bestResult;
            }

            if (key < node.val && node.left != null) {
                return findCLosestBST(node.left, key, minDiff, bestResult);
            } else if (key > node.val && node.right != null) {
                return findCLosestBST(node.right, key, minDiff, bestResult);
            } else {
                return bestResult;
            }
        }

        class SerializableTree {
            public String value;
            public ArrayList<SerializableTree> children = new ArrayList<>();
            int childCount;

            public SerializableTree() {
            }

            public SerializableTree(String val) {
                value = val;
            }

            public void addChild(SerializableTree child) {
                children.add(child);
            }

            public SerializableTree(String val, SerializableTree[] childs) {
                value = val;

                for (int i = 0; i < childs.length; i++) {
                    children.add(childs[i]);
                }
            }

            public String serialize(SerializableTree root) {
                StringBuilder serialized = new StringBuilder();
                Queue<SerializableTree> queue = new LinkedList<SerializableTree>();
                queue.offer(root);

                while (!queue.isEmpty()) {
                    SerializableTree node = queue.poll();
                    int childrenCount = node.children.size();

                    serialized.append(node.value);
                    serialized.append(",");
                    serialized.append(childrenCount);
                    serialized.append("#");

                    for (int i = 0; i < childrenCount; i++) {
                        SerializableTree child = node.children.get(i);
                        queue.offer(child);
                    }
                }

                return serialized.toString();
            }

            public SerializableTree deserialize(String serialized) {

                Queue<SerializableTree> queue = new LinkedList<SerializableTree>();
                String[] bfsNodes = serialized.split("#");
                String rootSer[] = bfsNodes[0].trim().split(",");

                SerializableTree root = new SerializableTree();
                root.value = rootSer[0].trim();
                root.childCount = Integer.parseInt(rootSer[1].trim());
                queue.offer(root);

                int serIndex = 1;
                while (!queue.isEmpty()) {
                    SerializableTree node = queue.poll();

                    for (int i = 0; i < node.childCount; i++) {
                        String childSer[] = bfsNodes[serIndex + i].trim().split(",");

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
        
        public TreeNode[] randomKSampleTreeNode(TreeNode root, int k) {
            Queue<TreeNode> q = new ArrayDeque<>();
            TreeNode[] reservoir = new TreeNode[k];
            q.add(root);
            int i = 0;
            
            while(!q.isEmpty()) {
                TreeNode node = q.poll();
                
                // add to reservoir as long as it is not full (size k)
                if(i < k) {
                    reservoir[i++] = node;
                }
                // if reservoir is full then take one sample (random) from index 0 to i 
                // if the random index is within reservoir (< k) then replace the slot with this new node
                else {
                    int index = (int) Math.random()*(i + 1);
                    if(index < k) {
                        reservoir[i++] = node;
                    }
                }
                
                if(node.left != null)
                    q.add(node.left);
                if(node.right != null)
                    q.add(node.right);
            }
            
            return reservoir;
        }

        public TreeNode[] randomKSampleTreeNode1(TreeNode root, int k) {
            TreeNode[] reservoir = new TreeNode[k];
            Queue<TreeNode> queue = new LinkedList<TreeNode>();
            queue.offer(root);
            int index = 0;

            // copy first k elements into reservoir
            while (!queue.isEmpty() && index < k) {
                TreeNode node = queue.poll();
                reservoir[index++] = node;
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }

            // for index k+1 to the last node of the tree select random index from (0 to
            // index)
            // if random index is less than k than replace reservoir node at this index by
            // current node
            while (!queue.isEmpty()) {
                TreeNode node = queue.poll();
                int j = (int) Math.floor(Math.random() * (index + 1));
                index++;

                if (j < k) {
                    reservoir[j] = node;
                }

                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }

            return reservoir;
        }
    }
    

    class BinarryTreeFlattenReconstruct {

        public ListNode flattenBT2SLLRecursive(TreeNode root) {
            if (root == null) {
                return null;
            }

            ListNode head = new ListNode(root.val);
            ListNode left = null;
            ListNode right = null;
            if (root.left != null) {
                left = flattenBT2SLLRecursive(root.left);
            }
            if (root.right != null) {
                right = flattenBT2SLLRecursive(root.right);
            }

            head.next = right;
            ListNode predecessor = left;
            while (predecessor != null && predecessor.next != null) {
                predecessor = predecessor.next;
            }

            if (predecessor != null) {
                predecessor.next = head;
                head = left;
            }

            return head;
        }

        public void flattenBT2SLLRecursiveFaster(TreeNode root, LinkedList<TreeNode> head) {
            if (root == null) {
                return;
            }

            flattenBT2SLLRecursiveFaster(root.left, head);
            head.add(root);
            flattenBT2SLLRecursiveFaster(root.right, head);
        }

        //
        public ListNode flattenBT2SLLIterative(TreeNode root) {
            if (root == null) {
                return null;
            }

            ListNode head = null;
            ;
            ListNode iterator = head;
            TreeNode cur = root;
            TreeNode pre = null;
            while (cur != null) {
                // if no left subtree the visit right subtree right away after printing current
                // node
                if (cur.left == null) {
                    if (head == null) {
                        head = new ListNode(cur.val);
                        iterator = head;
                    } else {
                        iterator.next = new ListNode(cur.val);
                        iterator = iterator.next;
                    }
                    System.out.print(cur.val + ", ");
                    cur = cur.right;
                } else {
                    // otherwise we will traverse the left subtree and come back to current
                    // node by using threaded pointer from predecessor of current node
                    // first find the predecessor of cur
                    pre = cur.left;
                    while (pre.right != null && pre.right != cur) {
                        pre = pre.right;
                    }

                    // threaded pointer not added - add it and go to left subtree to traverse
                    if (pre.right == null) {
                        pre.right = cur;
                        cur = cur.left;
                    } else {
                        // we traversed left subtree through threaded pointer and reached cur again
                        // so revert the threaded pointer and print out current node before traversing
                        // right subtree
                        pre.right = null;
                        if (head == null) {
                            head = new ListNode(cur.val);
                            iterator = head;
                        } else {
                            iterator.next = new ListNode(cur.val);
                            iterator = iterator.next;
                        }
                        System.out.print(cur.val + ", ");
                        // now traverse right subtree
                        cur = cur.right;
                    }
                }
            }

            return head;
        }

        // reusing the tree nodes
        public TreeNode flattenBT2SLLIterativeInplace(TreeNode root) {
            if (root == null) {
                return null;
            }
            TreeNode iterator = null;
            TreeNode head = null;
            TreeNode cur = root;
            TreeNode pre = null;
            while (cur != null) {
                // if no left subtree the visit right subtree right away after printing current
                // node
                if (cur.left == null) {
                    if (head == null) {
                        head = cur;
                        iterator = head;
                    } else {
                        iterator.right = cur;
                        iterator = iterator.right;
                    }
                    System.out.print(cur.val + ", ");
                    cur = cur.right;
                } else {
                    // otherwise we will traverse the left subtree and come back to current
                    // node by using threaded pointer from predecessor of current node
                    // first find the predecessor of cur
                    pre = cur.left;
                    while (pre.right != null && pre.right != cur) {
                        pre = pre.right;
                    }

                    // threaded pointer not added - add it and go to left subtree to traverse
                    if (pre.right == null) {
                        pre.right = cur;
                        cur = cur.left;
                    } else {
                        // we traversed left subtree through threaded pointer and reached cur again
                        // so revert the threaded pointer and print out current node before traversing
                        // right subtree
                        pre.right = null;
                        if (head == null) {
                            head = cur;
                            iterator = head;
                        } else {
                            iterator.right = cur;
                            iterator = iterator.right;
                        }
                        System.out.print(cur.val + ", ");
                        // now traverse right subtree
                        cur = cur.right;
                    }
                }
            }

            return head;
        }

        // The idea of inplace is to use tree's left and right pointer to connect linked
        // list nodes
        // It is possible because once we visit childs of a node we don't actually need
        // left/right
        // pointers anymore. So, we can reuse them for pointing prev/next nodes.
        public void flattenBT2LLPreOrderInplace(TreeNode root) {
            if (root == null) {
                return;
            }

            TreeNode cur = root;
            while (cur != null) {
                // if cur has a left child then we would like to flatten the left subtree
                // recursively
                // and put then under right child of cur so that we get a flatten list by right
                // pointer to traverse
                // We put left subtree first and then right subtree
                if (cur.left != null) {
                    // As we will put flattened left subtree to the right pointer of cur so
                    // before doing that we need to point the last (rightmost) node of flattened
                    // left subtree
                    // to point to right subtree (if it exists)
                    if (cur.right != null) {
                        TreeNode last = cur.left;
                        while (last.right != null) {
                            last = last.right;
                        }

                        last.right = cur.right;
                    }

                    // now update next (right) pointer of cur node to flattened left subtree
                    cur.right = cur.left;
                    cur.left = null;// Single Linked list - so no prev pointer
                }
                // if thers is no left subtree the we directly go to right subtree and flatten
                // it out
                else {
                    cur = cur.right;
                }
            }
        }
 
        public TreeNode sortedListToBST(ListNode head) {
            return sortedListToBSTHelper(head);
        }
        
        public TreeNode sortedListToBSTHelper(ListNode head) {
            if (head == null)
                return null;
            if (head.next == null){
                return new TreeNode(head.val);
            }

            // cut the list in middle
            ListNode prev = null, slow = head, fast = head;

            while (fast != null && fast.next != null) {
                prev = slow;
                slow = slow.next;
                fast = fast.next.next;
            }

            prev.next = null;
            
            // slow is the root 
            TreeNode root = new TreeNode(slow.val);
            // recursively compute left ans right subtree
            root.left = sortedListToBSTHelper(head);
            root.right = sortedListToBSTHelper(slow.next);

            return root;
        }

        public TreeNode convertList2BTreeRecursive(ListNode head) {
            int n = 0;
            ListNode temp = head;
            while (temp != null) {
                n++;
                temp = temp.next;
            }

            return convertList2BTreeRecursive(head, 0, n - 1);
        }

        // works for sorted/unsorted single/double linked list and for both BT and BST
        public TreeNode convertList2BTreeRecursive(ListNode h, int start, int end) {
            if (start > end) {
                return null;
            }

            // keep halving
            int mid = (start) + (end - start) / 2;

            // build left subtree
            TreeNode left = convertList2BTreeRecursive(h, start, mid - 1);
            // build root from current node
            TreeNode root = new TreeNode(h.val);
            // update left
            root.left = left;
            // build right subtree - first we need to increment head pointer
            // java pass objects by reference , so we can't just do h = h.next
            // instead we can update the head by value of head.next
            // head = head.next;
            if (h.next != null) {
                h.val = h.next.val;
                h.next = h.next.next;
                root.right = convertList2BTreeRecursive(h, mid + 1, end);
            }

            return root;
        }

        public TreeNode convertList2BTreeIterative(ListNode head) {
            if (head == null) {
                return null;
            }
            Queue<TreeNode> queue = new LinkedList<>();
            TreeNode root = new TreeNode(head.val);
            head = head.next;
            queue.offer(root);

            TreeNode node = null;
            while (!queue.isEmpty()) {
                node = queue.poll();
                if (head != null) {
                    node.left = new TreeNode(head.val);
                    head = head.next;
                    queue.offer(node.left);
                }
                if (head != null) {
                    node.right = new TreeNode(head.val);
                    head = head.next;
                    queue.offer(node.right);
                }
            }

            return root;
        }

        public ListNode convertList2BTreeInplace(ListNode head) {
            int n = 0;
            ListNode temp = head;
            while (temp != null) {
                n++;
                temp = temp.next;
            }

            return convertList2BTreeInplace(head, 0, n - 1);
        }

        // works in place for sorted/unsorted single/double linked list and for both BT
        // and BST
        public ListNode convertList2BTreeInplace(ListNode h, int start, int end) {
            if (start > end) {
                return null;
            }

            // keep halving
            int mid = (start) + (end - start) / 2;

            // build left subtree
            ListNode left = convertList2BTreeInplace(h, start, mid - 1);
            // build root from current node
            ListNode root = new ListNode(h.val);// h;
            // update left
            root.prev = left;
            // build right subtree - first we need to increment head pointer
            // java pass objects by reference , so we can't just do h = h.next
            // instead we can update the head by value of head.next
            // head = head.next;
            if (h.next != null) {
                h.val = h.next.val;
                h.next = h.next.next;
                root.next = convertList2BTreeInplace(h, mid + 1, end);
            }

            return root;
        }

        public TreeNode flattenBST2SortedDLLInplace(TreeNode root) {
            if (root == null) {
                return null;
            }

            // convert left subtree to DLL and connect last node (=predecessor of current
            // root) to current root
            if (root.left != null) {
                // convert left subtree
                TreeNode left = flattenBST2SortedDLLInplace(root.left);

                // find last node of the left DLL
                while (left.right != null) {
                    left = left.right;
                }

                // connect left DLL to root
                left.right = root;
                root.left = left;
            }
            // convert right subtree to DLL and connect root to the first node (=successor
            // of current root)
            if (root.right != null) {
                // convert left subtree
                TreeNode right = flattenBST2SortedDLLInplace(root.right);

                // find first node of the left DLL
                while (right.left != null) {
                    right = right.left;
                }

                // connect left DLL to root
                right.left = root;
                root.right = right;
            }

            return root;
        }

        public TreeNode flattenBST2SortedCircularDLLInplace(TreeNode root) {
            if (root == null) {
                return null;
            }

            // recursively divide it into left and right subtree until we get a leaf node
            // which
            // can be a stand alone doubly circular linked list by doing some simple pointer
            // manipulation
            TreeNode left = flattenBST2SortedCircularDLLInplace(root.left);
            TreeNode right = flattenBST2SortedCircularDLLInplace(root.right);

            // Let's first convert the root node into a stand alone circular DLL - just make
            // it a self loop
            root.right = root;
            root.left = root;

            // We have now sublist on the left of root and sublist on the right of root.
            // So, we just need to append left, root, and right sublists in the respective
            // order
            left = concatCircularDLL(left, root);
            left = concatCircularDLL(left, right);

            return left;
        }

        public void joinCircularNodes(TreeNode node1, TreeNode node2) {
            node1.right = node2;
            node2.left = node1;
        }

        // concats head2 list to the end of head1 list
        public TreeNode concatCircularDLL(TreeNode head1, TreeNode head2) {
            if (head1 == null) {
                return head2;
            }
            if (head2 == null) {
                return head1;
            }
            // in order to concat two circular list we need to
            // 1. join tail1 and head2 to append list2 to at the end of list1
            // 2. join tail2 and head1 to make it circular
            TreeNode tail1 = head1.left;
            TreeNode tail2 = head2.left;

            // join tail1 and head2 to append list2 to at the end of list1
            joinCircularNodes(tail1, head2);
            // join tail2 and head1 to make it circular
            joinCircularNodes(tail2, head1);

            return head1;
        }
    }

    class LinkedListOps {

        public ListNode deleteNodeWithHighOnRight(ListNode head) {
            ListNode temp = null;
            ListNode newHead = null;
            ListNode prev = head;
            while (head != null && head.next != null) {
                temp = head.next;

                if (temp.val > head.val) {
                    prev.next = temp;
                } else {
                    if (newHead == null) {
                        newHead = head;
                    }
                    prev = head;
                }

                head = head.next;
            }

            return newHead;
        }

        public ListNode removeNthFromEnd(ListNode head, int n) {
            if (head == null) {
                return null;
            }

            ListNode tail = head;
            while (tail != null && --n > 0) {
                tail = tail.next;
            }

            ListNode tempHead = head;
            ListNode prev = null;
            while (tail != null && tail.next != null) {
                tail = tail.next;
                prev = tempHead;
                tempHead = tempHead.next;
            }

            if (tempHead != null) {
                if (prev != null)
                    prev.next = tempHead.next;
                else {
                    head = tempHead.next;
                }
            }

            return head;
        }

        public boolean detectCycle(ListNode head) {
            ListNode slow = head;
            ListNode fast = head.next;

            while (slow != null && fast != null && slow != fast) {
                if (fast.next == null) {
                    break;
                }
                slow = slow.next;
                fast = fast.next.next;
            }

            if (slow != null && fast != null && slow == fast) {
                return true;
            }

            return false;
        }

        public void removeCycle(ListNode head) {
            ListNode slow = head;
            ListNode fast = head.next;

            while (fast != null && fast.next != null) {
                if (slow == fast) {
                    break;
                }
                slow = slow.next;
                fast = fast.next.next;
            }

            if (slow == fast) {
                slow = head;
                while (slow != fast.next) {
                    slow = slow.next;
                    fast = fast.next;
                }

                fast.next = null;
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

        public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
            ListNode headA1 = headA;
            ListNode headB1 = headB;

            while (headA1 != headB1) {
                headA1 = headA1 != null ? headA1.next : headB;
                headB1 = headB1 != null ? headB1.next : headA;
            }

            return headA1;
        }

        public ListNode reverseList(ListNode head) {
            ListNode newHead = null;
            ListNode cur = head;
            ListNode temp = null;

            while (cur != null) {
                temp = cur;
                cur = cur.next;
                temp.next = newHead;
                newHead = temp;
            }

            return newHead;
        }

        public ListNode reverse(ListNode head, ListNode reversed) {
            if (head == null) {
                return reversed;
            }

            ListNode current = head;
            head = head.next;
            current.next = reversed;

            return reverse(head, current);
        }
        
        public ListNode reverseBetween(ListNode head, int m, int n) {
            ListNode dummyHead = new ListNode(-1);
            dummyHead.next = head;
            ListNode tail = head;
            ListNode left = dummyHead;
            
            // set two pointers - left and right
            // left points to the prev node of the mth node (head)
            // right points to the next node node if of the nth node (tail)
            // both can be null
            int i = 1;
            while(i < n && tail != null){
                tail = tail.next;
                if(i < m){
                    left = head;
                    head = head.next;
                }
                i++;
            }

            // now reverse between the head and tail
            ListNode right = tail == null ? null : tail.next;
            // cut of the tail as tail is now pointed by right
            tail.next = null;
            // reverse between head and tail
            // rerversed list should point to the tail
            // so use the right pointer as the new revesed head
            while(head != null){
                ListNode cur = head;
                head = head.next;
                cur.next = right;
                right = cur;
            }
            
            // connect the left list with the right list
            left.next = right;
            
            return dummyHead.next;
        }
        
        public ListNode reverseK(ListNode head, ListNode tail, ListNode reversed, int k, int count) {
            if (head == null) {
                return reversed;
            }

            if (count == k) {
                tail.next = reverseK(head, null, null, k, 0);
                return reversed;
            } else {
                ListNode current = head;
                if (reversed == null) {
                    tail = current;
                }
                head = head.next;
                current.next = reversed;
                return reverseK(head, tail, current, k, count + 1);
            }
        }

        public ListNode swapPairs(ListNode head) {
            return reverseKGroup(head, 2);
        }

        public ListNode reverseKGroup(ListNode head, int k) {
            if (head == null || head.next == null) {
                return head;
            }

            ListNode prevHead = head;
            ListNode reversed = null;
            ListNode temp = null;
            int count = 0;

            // handle right most part of list of size less than k
            temp = head;
            while (temp != null && count < k) {
                count++;
                temp = temp.next;
            }
            if (count < k) {
                return head;
            }

            temp = null;
            count = 0;
            // reverse k nodes startifrom head
            // 1 -> 2 -> 3 -> 4 -> 5 -> null , and k = 3
            // then temp pointing at 4 -> 5 -> null
            // so revert the 1 -> 2 -> 3 portion
            // result would be 3 -> 2 -> 1 -> 4 -> 5 -> null
            //
            while (head != null && count < k) {
                temp = head.next;
                head.next = reversed;
                reversed = head;
                head = temp;
                count++;
            }

            // c
            if (prevHead != null) {
                prevHead.next = reverseKGroup(head, k);
            }

            return reversed;
        }
        
        // cut the list at kth node 
        // then traverse right part and put in front of first
        public ListNode rotateRight(ListNode head, int k) {
            if (k == 0 || head == null || head.next == null) {
                return head;
            }

            int n = 0;
            ListNode p1 = head;
            // tail1 is the tail of the list 
            ListNode tail = head;
            while (p1 != null) {
                n++;
                tail = p1;
                p1 = p1.next;
            }

            k = k % n;
            if (k == 0) {
                return head;
            }

            // find the head of the right part
            // it is kth node from the last
            // we already know the length so, just use counter
            p1 = head;
            ListNode prev = null;
            // kth node from the last is n-kth node from begining
            k = n-k;
            while (k-- > 0) {
                prev = p1;
                p1 = p1.next;
            }
            
            // now move right part in front of left part
            tail.next = head;
            head = p1;
            prev.next = null;

            return head;
        }
        
        public ListNode rotateListRight(ListNode head, int k) {
            if (k == 0 || head == null || head.next == null) {
                return head;
            }

            int n = 0;
            ListNode slow = head;
            while (slow != null) {
                n++;
                slow = slow.next;
            }

            k = k % n;
            if (k == 0) {
                return head;
            }

            // now split at k from end
            slow = head;
            ListNode fast = head;
            while (k-- > 0) {
                fast = fast.next;
            }

            // use two pointer to split
            ListNode prevSlow = null;
            ListNode prevFast = null;
            while (fast != null) {
                prevSlow = slow;
                slow = slow.next;
                prevFast = fast;
                fast = fast.next;
            }

            prevFast.next = head;
            prevSlow.next = null;
            head = slow;

            return head;
        }

        public ListNode mergeSortedLists(ListNode a, ListNode b) {
            if (a == null) {
                return b;
            }
            if (b == null) {
                return a;
            }

            ListNode merged = null;

            merged = b;
            if (a.val > b.val) {
                merged.next = mergeSortedLists(a, b.next);
            } else {
                merged = a;
                merged.next = mergeSortedLists(a.next, b);
            }

            return merged;
        }

        public ListNode merge(ListNode l1, ListNode l2) {
            if (l1 == null || l2 == null) {
                return l1 == null ? l2 : l1;
            }

            ListNode dummy = new ListNode(0), cur = l1;

            while (l1 != null && l2 != null) {
                if (l1.val <= l2.val) {
                    cur.next = l1;
                    l1 = l1.next;
                } else {
                    cur.next = l2;
                    l2 = l2.next;
                }
            }

            if (l1 != null) {
                cur.next = l1;
            } else if (l2 != null) {
                cur.next = l2;
            }

            return dummy.next;
        }

        // sort a list
        public ListNode MergeSortList(ListNode head) {
            if (head == null || head.next == null)
                return head;

            // cut the list in middle
            ListNode prev = null, slow = head, fast = head;

            while (fast != null && fast.next != null) {
                prev = slow;
                slow = slow.next;
                fast = fast.next.next;
            }

            prev.next = null;

            ListNode left = MergeSortList(head);
            ListNode right = MergeSortList(slow);

            return mergeSortedLists(left, right);
        }

        // sort a list of sorted lists
        public ListNode mergeKLists(ListNode[] lists) {
            if (lists == null || lists.length == 0) {
                return null;
            }
            if (lists.length == 1) {
                return lists[0];
            }

            return mergeKListsDivide(lists, 0, lists.length - 1);
        }

        public ListNode mergeKListsDivide(ListNode[] lists, int start, int end) {
            if (start == end) {
                return lists[start];
            }

            int mid = start + (end - start) / 2;
            ListNode left = mergeKListsDivide(lists, start, mid);
            ListNode right = mergeKListsDivide(lists, mid + 1, end);

            return mergeSortedLists(left, right);
        }

        public ListNode splitLinkedListNode(ListNode head, int n) {
            ListNode slow = head;
            ListNode fast = head;
            ListNode prev = head;

            while (fast != null && slow != null) {
                int count = 0;
                prev = slow;
                slow = slow.next;
                // for every one move of slow we move fast n times
                while (count < n && fast != null) {
                    fast = fast.next;
                    count++;
                }

                if (slow == fast) {
                    return null;
                }
            }

            if (prev != null) {
                prev.next = null;
            }

            return slow;
        }
        
        /**
         * Given a (singly) linked list with head node root, write a function to split the linked list 
         * into k consecutive linked list "parts". 
         * 
         * The length of each part should be as equal as possible: no two parts should have a size 
         * differing by more than 1. This may lead to some parts being null.
         * Input: root = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], k = 3
         * Output: [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]
         * 
         * Input: root = [1, 2, 3], k = 5
         * Output: [[1],[2],[3],[],[]]
         * 
         * @param root
         * @param k
         * @return
         */
        public ListNode[] splitListToParts(ListNode root, int k) {
            ListNode[] buckets = new ListNode[k];
            
            ListNode head = root;
            int n = 0;
            while(head != null) {
                head = head.next;
                n++;
            }
            
            if(n == 0){
                return buckets;
            }
            
            int bucketSize = n/k;
            int bucketsWithExtraOne = n%k;
            
            ListNode cur = root, prev = null;
            for(int i = 0; i < k; i++){
                buckets[i] = cur;
                // now move head to next bucket head
                int fillSize = bucketSize+ (bucketsWithExtraOne-- > 0 ? 1 : 0);
                for(int j = 0; cur != null && j < fillSize; j++){
                    prev = cur;
                    cur = cur.next;
                }
                prev.next = null;
            }
            
            return buckets;
        }

        class SolutionMergeKListWithPQ {
            class Cell implements Comparable<Cell> {
                public int val;
                public int index;

                public Cell(int val, int index) {
                    this.val = val;
                    this.index = index;
                }

                @Override
                public int compareTo(Cell o) {
                    if (this.val == o.val) {
                        return Integer.compare(this.index, o.index);
                    } else {
                        return Integer.compare(this.val, o.val);
                    }
                }
            }

            public ListNode mergeKLists(ListNode[] lists) {
                if (lists == null || lists.length == 0) {
                    return null;
                }
                if (lists.length == 1) {
                    return lists[0];
                }

                PriorityQueue<Cell> pq = new PriorityQueue<>(lists.length);
                ListNode res = null;
                ListNode resIterator = null;
                for (int i = 0; i < lists.length; i++) {
                    if (lists[i] != null) {
                        pq.offer(new Cell(lists[i].val, i));
                        lists[i] = lists[i].next;
                    }
                }

                while (!pq.isEmpty()) {
                    Cell c = pq.poll();
                    ListNode node = new ListNode(c.val, null);
                    if (res == null) {
                        res = node;
                        resIterator = res;
                    } else {
                        resIterator.next = node;
                        resIterator = resIterator.next;
                    }

                    if (lists[c.index] != null) {
                        pq.offer(new Cell(lists[c.index].val, c.index));
                        lists[c.index] = lists[c.index].next;
                    }
                }

                return res;
            }
        }
    }

    class PermutationCombinatons {

        public List<List<Integer>> permute(int[] nums) {
            List<List<Integer>> res = new ArrayList<>();
            Set<Integer> visited = new HashSet<>();
            permutation(nums, new LinkedList<>(), res, visited);
            return res;
        }

        public void permutation(int[] nums, LinkedList<Integer> cur, List<List<Integer>> res, Set<Integer> visited) {
            if (cur.size() == nums.length) {
                res.add(new LinkedList<>(cur));
            } else {
                for (int i = 0; i < nums.length; i++) {
                    if (visited.contains(nums[i])) {
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

        public void permutationUnique(int[] nums, LinkedList<Integer> cur, List<List<Integer>> res,
                Set<Integer> visited) {
            if (cur.size() == nums.length) {
                res.add(new LinkedList<>(cur));
            } else {
                for (int i = 0; i < nums.length; i++) {
                    if (visited.contains(i) || (i > 0 && nums[i] == nums[i - 1] && !visited.contains(i - 1))) {
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

        public void allCasePermutataion(String str, char[] cur, int i, Set<String> res) {
            if(i == str.length()){
                res.add(new String(cur));
                return;
            }

            // skip permutations for letterrs
            if(Character.isLetter(cur[i])){
                cur[i] = Character.toLowerCase(cur[i]);
                allCasePermutataion(str, cur, i+1, res);

                cur[i] = Character.toUpperCase(cur[i]);
                allCasePermutataion(str, cur, i+1, res);
            }
            else{
                allCasePermutataion(str, cur, i+1, res);
            }
        }

        public List<List<Integer>> combine(int n, int k) {
            List<List<Integer>> result = new ArrayList<>();
            combination(n, k, new LinkedList<Integer>(), 0, result);
            return result;
        }

        public void combination(int n, int k, LinkedList<Integer> cur, int start, List<List<Integer>> result) {
            if (cur.size() == k) {
                result.add(new ArrayList<>(cur));
            } else {
                for (int i = start; i < n; i++) {
                    cur.add(i + 1);
                    combination(n, k, cur, i + 1, result);
                    cur.removeLast();
                }
            }
        }

        // can reuse the numbers
        public List<List<Integer>> combinationSum(int[] candidates, int target) {
            List<List<Integer>> result = new ArrayList<>();
            Arrays.sort(candidates);
            combinationSumHelper(candidates, target, 0, new LinkedList<>(), 0, result);
            return result;
        }

        public void combinationSumHelper(int[] nums, int targetSum, int curSum, LinkedList<Integer> cur, int start,
                List<List<Integer>> result) {
            if (curSum == targetSum) {
                result.add(new ArrayList<>(cur));
            }
            // prune unreachable paths
            else if (curSum > targetSum) {
                return;
            } else {
                for (int i = start; i < nums.length; i++) {
                    cur.add(nums[i]);
                    combinationSumHelper(nums, targetSum, curSum + nums[i], cur, i, result); // i to reuse the same
                                                                                             // numbers
                    cur.removeLast();
                }
            }
        }

        // use each number once only and no duplicate in the answer
        public List<List<Integer>> combinationSum2(int[] candidates, int target) {
            List<List<Integer>> result = new ArrayList<>();
            Arrays.sort(candidates);
            combinationSumHelper2(candidates, target, 0, new LinkedList<>(), 0, result);
            return result;
        }

        public void combinationSumHelper2(int[] nums, int targetSum, int curSum, LinkedList<Integer> cur, int start,
                List<List<Integer>> result) {
            if (curSum == targetSum) {
                result.add(new ArrayList<>(cur));
            }
            // prune unreachable paths
            else if (curSum > targetSum) {
                return;
            } else {
                for (int i = start; i < nums.length; i++) {
                    // skip duplicates
                    if (i == start || (i > start && nums[i] != nums[i - 1])) {
                        cur.add(nums[i]);
                        combinationSumHelper2(nums, targetSum, curSum + nums[i], cur, i + 1, result);// i+1 to use one
                                                                                                     // number just
                                                                                                     // once.
                        cur.removeLast();
                    }
                }
            }
        }

        /**
         * Given an integer array, your task is to find all the different possible increasing subsequences 
         * of the given array, and the length of an increasing subsequence should be at least 2.

            Example:
            
            Input: [4, 6, 7, 7]
            Output: [[4, 6], [4, 7], [4, 6, 7], [4, 6, 7, 7], [6, 7], [6, 7, 7], [7,7], [4,7,7]]

         * @param nums
         * @return
         */
        public List<List<Integer>> findIncreasingSubsequences(int[] nums) {
            List<List<Integer>> res = new ArrayList<>();

            combinationForLis(nums, 0, new LinkedList<>(), res);

            return res;
        }

        private void combinationForLis(int nums[], int start, LinkedList<Integer> cur, List<List<Integer>> res) {
            if (cur.size() > 1) {
                res.add(new ArrayList<>(cur));
            }

            Set<Integer> visited = new HashSet<>();
            for (int i = start; i < nums.length; i++) {
                if (visited.contains(nums[i]))
                    continue;
                if (cur.isEmpty() || (cur.peekLast() <= nums[i])) {
                    cur.add(nums[i]);
                    visited.add(nums[i]);
                    combinationForLis(nums, i + 1, cur, res);
                    cur.removeLast();
                }
            }
        }

        public List<List<Integer>> subsets(int[] nums) {
            List<List<Integer>> result = new ArrayList<>();
            subsetHelper(nums, new LinkedList<Integer>(), 0, result);
            return result;
        }

        public void subsetHelper(int[] nums, LinkedList<Integer> cur, int start, List<List<Integer>> result) {
            result.add(new ArrayList<>(cur));

            for (int i = start; i < nums.length; i++) {
                // skip dups
                if (i == start || (i > start && nums[i] != nums[i - 1])) {
                    cur.add(nums[i]);
                    subsetHelper(nums, cur, i + 1, result);// i+1 to take one element just once in the current recursion
                                                           // path
                    cur.removeLast();
                }
            }
        }

        public void permList(List<List<String>> input, String[] cur, int index, List<List<String>> result) {
            if (index == input.size()) {
                result.add(Arrays.asList(cur.clone()));
            } else {
                // for each candidate in current position (index) recurse to construct solution
                List<String> cands = input.get(index);
                for (int i = 0; i < cands.size(); i++) {
                    // add current candidate to temp solution
                    cur[index] = cands.get(i);
                    permList(input, cur, index + 1, result);
                    // backtrack
                    cur[index] = null;
                }
            }
        }

        public String[] map = new String[] { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };

        public List<String> letterCombinations(String digits) {
            List<String> result = new ArrayList<>();
            if (digits == null || digits.isEmpty()) {
                return result;
            }
            permList(digits, new StringBuilder(), 0, result);
            return result;
        }

        public void permList(String input, StringBuilder cur, int index, List<String> result) {
            if (index == input.length()) {
                result.add(cur.toString());
            } else {
                // for each candidate in current position (index) recurse to construct solution
                String cands = map[input.charAt(index) - '0'];
                for (int i = 0; i < cands.length(); i++) {
                    // add current candidate to temp solution
                    cur.append(cands.charAt(i));
                    permList(input, cur, index + 1, result);
                    // backtrack
                    cur.setLength(cur.length() - 1);
                }
            }
        }

        // all valid combinations of k numbers that sum up to n such that the following
        // conditions are true:
        // Only numbers 1 through 9 are used.
        // Each number is used at most once.
        public List<List<Integer>> combinationSum3(int k, int n) {
            List<List<Integer>> result = new ArrayList<>();
            combinationSumHelper3(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, k, n, 0, new LinkedList<>(), 0, result);
            return result;
        }

        public void combinationSumHelper3(int[] nums, int k, int targetSum, int curSum, LinkedList<Integer> cur,
                int start, List<List<Integer>> result) {
            if (cur.size() == k && curSum == targetSum) {
                result.add(new ArrayList<>(cur));
            }
            // prune unreachable paths
            else if (curSum > targetSum) {
                return;
            } else {
                for (int i = start; i < nums.length; i++) {
                    // skip duplicates
                    if (i == start || (i > start && nums[i] != nums[i - 1])) {
                        cur.add(nums[i]);
                        combinationSumHelper3(nums, k, targetSum, curSum + nums[i], cur, i + 1, result);// i+1 to use
                                                                                                        // one number
                                                                                                        // just once.
                        cur.removeLast();
                    }
                }
            }
        }
    }

    class SubsetSum {

        public boolean isSubSetSum(final int[] set, final int sum) {
            final int m = set.length;
            final boolean[][] ssTable = new boolean[sum + 1][m + 1];

            // base cases: if m == 0 then no solution for any sum
            for (int i = 0; i <= sum; i++) {
                ssTable[i][0] = false;
            }
            // base case: if sum = 0 then there is only one solution for any input set: just
            // take none of each of the items.
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
    }

    class GraphTraversals {
        int[] vertices;
        Edge[][] edges;

        public List<Integer> shortestPath(int start, int end) {
            List<Integer> shortestPath = new LinkedList<>();
            int[] shortestPathLen = new int[vertices.length + 1];
            Arrays.fill(shortestPathLen, Integer.MAX_VALUE);

            int[] shortestPathParent = new int[vertices.length + 1];
            Arrays.fill(shortestPathParent, -1);

            boolean[] visited = new boolean[vertices.length + 1];
            Queue<Integer> queue = new LinkedList<>();
            queue.add(start);

            while (!queue.isEmpty()) {
                int u = queue.remove();
                visited[u] = true;

                for (Edge e : edges[u]) {
                    if (e != null) {
                        if ((shortestPathLen[u] + e.w) < shortestPathLen[e.v]) {
                            shortestPathLen[e.v] = shortestPathLen[u] + e.w;
                            shortestPathParent[e.v] = u;
                        }

                        if (!visited[e.v]) {
                            queue.add(e.v);
                        }
                    }
                }
            }

            int i = end;
            while (shortestPathParent[i] != -1) {
                shortestPath.add(0, vertices[i]);
                i = shortestPathParent[i];
            }
            shortestPath.add(0, vertices[start]);

            return shortestPath;
        }
        
        class Trie {

            /** Initialize your data structure here. */
            class Node{
                char c;
                boolean hasWord;
                Map<Character, Node> children;
                
                public Node(char c){
                    this.c = c;
                    this.hasWord = false;
                    this.children = new HashMap<>();
                }
            }
            
            Node root;
            public Trie() {
                this.root = new Node('\0');
            }
            
            /** Inserts a word into the trie. */
            public void insert(String word) {
                int n = word.length();
                Node parent = root;
                for(int i = 0; i < n; i++) {
                    char c = word.charAt(i);
                    Node child = parent.children.getOrDefault(c, new Node(c));
                    child.hasWord |= (i == (n-1));
                    parent.children.put(c, child);
                    parent = child;
                }
            }
            
            /** Returns if the word is in the trie. */
            public boolean search(String word) {
                int n = word.length();
                Node parent = root;
                for(int i = 0; i < n; i++) {
                    char c = word.charAt(i);
                    Node child = parent.children.get(c);
                    if(child != null){
                        parent = child;
                        if((i == n-1) && child.hasWord){
                            return true;
                        }
                    }
                    else{
                        break;
                    }
                }
                
                return false;
            }
            
            /** Returns if there is any word in the trie that starts with the given prefix. */
            public boolean startsWith(String prefix) {
                int n = prefix.length();
                Node parent = root;
                for(int i = 0; i < prefix.length(); i++) {
                    char c = prefix.charAt(i);
                    Node child = parent.children.get(c);
                    if(child != null){
                        parent = child;
                        if(i == (n-1)){
                            return true;
                        }
                    }
                    else{
                        break;
                    }
                }
                
                return false;
            }
        }
    }

    class WalkBFS {
        // find exit path to the door
        public void findExitWallsAndGates(int room[][]) {
            Queue<int[]> queue = new LinkedList<int[]>();
            int n = room.length;
            int m = room[0].length;
            // down, right, up, left
            int[][] dirs = new int[][] { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 } };

            for (int i = 0; i < room.length; i++) {
                for (int j = 0; j < room[0].length; j++) {
                    if (room[i][j] == 0) {
                        queue.offer(new int[] { i, j });
                    }
                }
            }

            // BFS search
            while (!queue.isEmpty()) {
                int[] pos = queue.poll();
                int r = pos[0];
                int c = pos[1];

                for (int[] dir : dirs) {
                    int i = r + dir[0];
                    int j = c + dir[1];

                    // prune the tree
                    if (i < 0 || j < 0 || i >= n || j >= m || room[i][j] <= room[r][c] + 1) {
                        continue;
                    }

                    room[i][j] = room[r][c] + 1;
                    queue.offer(new int[] { i, j });
                }
            }
        }
        
        public boolean isNavigable(final String src, final String dst, final Set<String> dictionary) {
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
        
        public int ladderLength(String src, String dst, List<String> wordList) {
            if (src.length() != dst.length()) {
                return 0;
            }
            if (src.equals(dst)) {
                return 1;

            }

            Set<String> dictionary = new HashSet<>(wordList);
            if(!dictionary.contains(dst)){
                return 0;
            }

            final Queue<String> q = new ArrayDeque<String>();
            // add root to queue
            q.add(src);
            // use dictionary itself as un-visited set
            dictionary.remove(src);
            int len = 1;
            while (!q.isEmpty()) {
                int size = q.size();
                // visit all nodes in the queue first for level first order
                for(int i = 0; i < size; i++){
                    String word = q.poll();
                    // remove this node from unvisites set
                    dictionary.remove(word);

                    // as we are diong breadth and level first 
                    // so first match would ne the shortest path
                    if(word.equals(dst)){
                        return len;
                    }

                    // if no match then visit each of the allowed transformed candidte
                    char[] chars = word.toCharArray();
                    for(int j = 0; j < chars.length; j++){
                        // for each position try to trnsform each of the 26 possible characters
                        char original = chars[j];
                        for(char c = 'a'; c <= 'z'; c++){
                            chars[j] = c;
                            String intermediate = new String(chars);

                            // visit BFS this intermediate word if
                            // it contains in dictionary and yet not visited
                            if(dictionary.contains(intermediate)){
                                q.add(intermediate);
                                dictionary.remove(intermediate);
                            }
                        }

                        // put back original character on the position
                        chars[j] = original;
                    }

                }

                len++;
            }

            return 0;
        }
        
        // using Dijkstra
        public int ladderLength2(String src, String dst, List<String> wordList) {
            if (src.length() != dst.length()) {
                return 0;
            }
            if (src.equals(dst)) {
                return 1;

            }

            Set<String> dictionary = new HashSet<>(wordList);
            if(!dictionary.contains(dst)){
                return 0;
            }
            dictionary.remove(src);

            Set<String> visited = new HashSet<>();
            Map<String, Integer> shortestPathLens = new HashMap<>();
            final Queue<String> q = new ArrayDeque<String>();
            // add root to queue
            q.add(src);
            shortestPathLens.put(src, 0);
            
            int minLen = Integer.MAX_VALUE;
            while (!q.isEmpty()) {
                String u = q.remove();
                visited.add(u);
                
                // visit all the edges
                for(String e : getLadderEdges(u, dictionary, visited)) {
                    if(visited.contains(e)) {
                        continue;
                    }
                    // Dijkstra's inequality - select this path if it is shorter
                    // notice the <= instead of < because we want all the paths with min length, not any one path
                    if(!shortestPathLens.containsKey(e) || (shortestPathLens.get(u)+1 <= shortestPathLens.get(e))) {
                        shortestPathLens.put(e, shortestPathLens.get(u)+1);
                    }
                    
                    // if the destination can be reached update the minLen
                    if(e.equals(dst)) {
                        minLen = Math.min(minLen, 1+shortestPathLens.get(dst));
                    }
                    
                    // visit the node if not visited yet
                    if(!visited.contains(e)) {
                        q.add(e);
                    }
                }
            }

            return minLen;
        }
        
        // send all the edges (all character mutations for all positions)
        private Set<String> getLadderEdges(String word, Set<String> dictionary, Set<String> visited) {
            Set<String> edges = new HashSet<>();
            
            for (int i = 0; i < word.length(); i++) {
                char[] candidateChars = word.toCharArray();
                // all possible words with current character variations
                for (char c = 'a'; c <= 'z'; c++) {
                    candidateChars[i] = c;
                    String candidate = new String(candidateChars);
                    
                    if (dictionary.contains(candidate) && !visited.contains(candidate)) {
                        edges.add(candidate);
                    }
                }
            }
            
            return edges;
        }
        
        /**
         * A gene string can be represented by an 8-character long string, with choices from "A", "C", "G", "T".
         * Suppose we need to investigate about a mutation (mutation from "start" to "end"), where ONE mutation is 
         * defined as ONE single character changed in the gene string.
         * For example, "AACCGGTT" -> "AACCGGTA" is 1 mutation.
         * Also, there is a given gene "bank", which records all the valid gene mutations. 
         * A gene must be in the bank to make it a valid gene string.
         * 
         * Now, given 3 things - start, end, bank, your task is to determine what is the minimum number of 
         * mutations needed to mutate from "start" to "end". If there is no such a mutation, return -1.
         * 
         * @param src
         * @param dst
         * @param bank
         * @return
         */
        public int minMutation(String src, String dst, String[] bank) {
            if (src.length() != dst.length() || bank.length == 0) {
                return -1;
            }
            if (src.equals(dst)) {
                return 0;

            }

            Set<String> dictionary = new HashSet<>(Arrays.asList(bank));
            if(!dictionary.contains(dst)){
                return -1;
            }
            dictionary.remove(src);

            Set<String> visited = new HashSet<>();
            Map<String, Integer> shortestPathLens = new HashMap<>();
            final Queue<String> q = new ArrayDeque<String>();
            // add root to queue
            q.add(src);
            shortestPathLens.put(src, -1);

            int minLen = Integer.MAX_VALUE;
            while (!q.isEmpty()) {
                String u = q.remove();
                visited.add(u);

                // visit all the edges
                for(String e : getLadderEdges(u, dictionary, visited)) {
                    if(visited.contains(e)) {
                        continue;
                    }
                    // Dijkstra's inequality - select this path if it is shorter
                    // notice the <= instead of < because we want all the paths with min length, not any one path
                    if(!shortestPathLens.containsKey(e) || (shortestPathLens.get(u)+1 <= shortestPathLens.get(e))) {
                        shortestPathLens.put(e, shortestPathLens.get(u)+1);
                    }

                    // if the destination can be reached update the minLen
                    if(e.equals(dst)) {
                        minLen = Math.min(minLen, 1+shortestPathLens.get(dst));
                        return minLen;
                    }

                    // visit the node if not visited yet
                    if(!visited.contains(e)) {
                        q.add(e);
                    }
                }
            }

            return minLen == Integer.MAX_VALUE? -1 : minLen;
        }
        
        public List<List<String>> wordLadderAll(Set<String> dictionary, String src, String dst) {
            if (src == null || dst == null || dictionary == null || src.isEmpty() || dst.isEmpty()
                    || dictionary.isEmpty()) {
                return Collections.emptyList();
            }
            // path from a node to its parent along the BFS traversal
            Map<String, Set<String>> shortestPathParent = new HashMap<String, Set<String>>();
            // level or length of a word appeared in the DAG
            Map<String, Integer> shortestPathLen = new HashMap<String, Integer>();
            // resulting shortest paths
            List<List<String>> shortestPaths = new ArrayList<>();
            // visited set
            Set<String> visited = new HashSet<>();
            
            // Queue to traverse in BFS
            Queue<String> queue = new ArrayDeque<String>();
            queue.add(src);
            shortestPathLen.put(src, 0);

            while (!queue.isEmpty()) {
                String u = queue.remove();
                visited.add(u);

                // traverse all the edges
                for (String e : getLadderEdges(u, dictionary, visited)) {
                    if (e != null) {
                        if(visited.contains(e)) {
                            continue;
                        }
                        // Dijkstra's inequality - select this path if it is shorter
                        // notice the <= instead of < because we want all the paths with min length, not any one path
                        if (!shortestPathLen.containsKey(e) || (shortestPathLen.get(u) + 1) <= shortestPathLen.get(e)) {
                            shortestPathLen.put(e, shortestPathLen.get(u) + 1);
                            // update parent
                            Set<String> p = shortestPathParent.getOrDefault(e, new HashSet<>());
                            p.add(u);
                            shortestPathParent.put(e, p);
                            continue;
                        }

                        // if not visited already then push to queue for visiting
                        if (!visited.contains(e)) {
                            queue.add(e);
                        }
                    }
                }
            }
            
            // run a DFS on the parent DAG from dest to source (reverse)
            getPathsDFS(dst, src, new LinkedList<>(), shortestPathParent, shortestPaths);
            
            return shortestPaths;
        }
        
        private void getPathsDFS(String src, String dst, LinkedList<String> cur, Map<String, Set<String>> parents, List<List<String>> res) {
            // as we are going from dest so add in front (reverse)
            cur.addFirst(src);
            if(src.equals(dst)) {
                res.add(new ArrayList<>(cur));
            }
            else {
                // visit all parents
                for(String p : parents.getOrDefault(src, new HashSet<>())) {
                    getPathsDFS(p, dst, cur, parents, res);
                }
            }
            // backtrack 
            cur.remove();
        }
    }

    class BackTrack {
        /**
         * A robot is located at the top-left corner of a m x n grid. The robot can only
         * move either down or right at any point in time. The robot is trying to reach
         * the bottom-right corner of the grid. How many possible unique paths are
         * there?
         * 
         * Soln: Since the robot can only move right and down, when it arrives at a
         * point, it either arrives from left or above. If we use dp[i][j] for the
         * number of unique paths to arrive at the point (i, j), then the state equation
         * is dp[i][j] = dp[i][j - 1] + dp[i - 1][j]. Moreover, we have the base cases
         * dp[0][j] = dp[i][0] = 1 for all valid i and j.
         * 
         * for (int i = 0; i < m; i++) { for (int j = 0; j < n; j++) { if(j > 0){
         * dp[i][j] = dp[i - 1][j] + dp[i][j - 1]; } } } return dp[m - 1][n - 1];
         * 
         * We can noticed that each time when we update dp[i][j], we only need dp[i -
         * 1][j] (at the previous row) and dp[i][j - 1] (at the current row).
         * 
         * So, we update current cell by taking decision from top cell dp[i - 1][j] and
         * left cell dp[i][j - 1] That is, we can reduce the memory usage to just two
         * rows (O(n)).
         * 
         * @param m
         * @param n
         * @return
         */
        public int uniquePaths(int m, int n) {
            if (m == 1 || n == 1) {
                return 1;
            }
            int[] dp = new int[n];

            dp[0] = 1;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (j > 0) {
                        // current cell = old top cell + old left cell
                        dp[j] = dp[j] + dp[j - 1];
                    }
                }
            }

            return dp[n - 1];
        }

        public int uniquePathsWithObstaclesDp(int[][] obstacleGrid) {
            if (obstacleGrid == null || obstacleGrid.length == 0) {
                return 0;
            }
            int[] dp = new int[obstacleGrid[0].length];

            dp[0] = 1;
            for (int i = 0; i < obstacleGrid.length; i++) {
                for (int j = 0; j < obstacleGrid[0].length; j++) {
                    // if obstacle then no solution through this cell
                    if (obstacleGrid[i][j] == 1) {
                        dp[j] = 0;
                    } else if (j > 0) {
                        // current cell = old top cell + old left cell
                        dp[j] = dp[j] + dp[j - 1];
                    }
                }
            }

            return dp[obstacleGrid[0].length - 1];
        }

        public int uniquePathsWithObstaclesBacktrack(int[][] obstacleGrid) {
            if (obstacleGrid == null || obstacleGrid.length == 0) {
                return 0;
            }
            int[][] count = new int[obstacleGrid.length][obstacleGrid[0].length];
            uniquePathsBacktrack(obstacleGrid, 0, 0, count);

            return count[obstacleGrid.length - 1][obstacleGrid[0].length - 1];
        }

        private void uniquePathsBacktrack(int[][] grid, int i, int j, int[][] count) {
            if (count[i][j] != 0) {
                count[i][j]++;
                return;
            }
            if ((i == grid.length - 1) && (j == grid[0].length - 1) && grid[i][j] == 0) {
                count[i][j]++;
                return;
            }
            if (grid[i][j] == 1) {
                return;
            }

            int dirs[][] = new int[][] { { 1, 0 }, { 0, 1 } };
            for (int[] dir : dirs) {
                if ((i + dir[0] < grid.length) && (j + dir[1] < grid[0].length)
                        && (grid[i + dir[0]][j + dir[1]] == 0)) {
                    uniquePathsBacktrack(grid, i + dir[0], j + dir[1], count);
                }
            }
        }

        /**
         * Given a 2D board and a word, find if the word exists in the grid. The word
         * can be constructed from letters of sequentially adjacent cells, where
         * "adjacent" cells are horizontally or vertically neighboring. The same letter
         * cell may not be used more than once.
         */
        public boolean existWordSearch(char[][] board, String word) {
            if (word == null || word.length() == 0 || board == null || board.length == 0) {
                return false;
            }
            for (int i = 0; i < board.length; i++) {
                for (int j = 0; j < board[i].length; j++) {
                    if (board[i][j] == word.charAt(0)) {
                        if (wordExistsBacktrack(board, i, j, word, 0)) {
                            return true;
                        }
                    }
                }
            }

            return false;
        }

        private boolean wordExistsBacktrack(char[][] board, int i, int j, String word, int k) {
            // base case
            if (k == word.length()) {
                return true;
            }

            // check for safety - prune the search
            if (i < 0 || i >= board.length || j < 0 || j >= board[i].length || board[i][j] != word.charAt(k)) {
                return false;
            }

            // set visited to avoid cycle, we can reuse the board
            board[i][j] ^= 256;
            boolean result = wordExistsBacktrack(board, i + 1, j, word, k + 1)
                    || wordExistsBacktrack(board, i - 1, j, word, k + 1)
                    || wordExistsBacktrack(board, i, j - 1, word, k + 1)
                    || wordExistsBacktrack(board, i, j + 1, word, k + 1);
            board[i][j] ^= 256;

            return result;
        }
        
        /**
         * Given a 2D board and a list of words from the dictionary, find all words in the board.
         * Each word must be constructed from letters of sequentially adjacent cell, 
         * where "adjacent" cells are those horizontally or vertically neighboring. 
         * The same letter cell may not be used more than once in a word.
         * 
         * Input: 
            board = [
              ['o','a','a','n'],
              ['e','t','a','e'],
              ['i','h','k','r'],
              ['i','f','l','v']
            ]
            words = ["oath","pea","eat","rain"]
            
            Output: ["eat","oath"]
         * 
         * @param board
         * @param words
         * @return
         */
        public List<String> findWords(char[][] board, String[] words) {
            if (words == null || words.length == 0 || board == null || board.length == 0) {
                return Collections.emptyList();
            }
            
            List<String> res = new ArrayList<>();
            TrieNode root = buildTrie(words);
            for (int i = 0; i < board.length; i++) {
                for (int j = 0; j < board[i].length; j++) {
                    findWordsBacktrack(board, i, j, root, res);
                }
            }

            return res;
        }

        private void findWordsBacktrack(char[][] board, int i, int j, TrieNode root, List<String> res) {
            // check for safety - prune the search
            if (i < 0 || i >= board.length || j < 0 || j >= board[i].length) {
                return;
            }
            
            char c = board[i][j];
            // base case
            if(c == '$' || root.childs[c-'a'] == null){
                return;
            }
            else{
                root = root.childs[c-'a'];
                if(root.hasWord){
                    res.add(root.word);
                    // do not add the same word to result
                    root.hasWord = false;
                }
            }
            
            // set visited to avoid cycle, we can reuse the board
            board[i][j] = '$';
            int n = board.length;
            int m = board[0].length;
            // backtrack
            if(i+1 < n) findWordsBacktrack(board, i + 1, j, root, res);
            if(i-1 >= 0) findWordsBacktrack(board, i - 1, j, root, res);
            if(j-1 >= 0) findWordsBacktrack(board, i, j - 1, root, res);
            if(j+1 < m) findWordsBacktrack(board, i, j + 1, root, res);
            board[i][j] = c;
        }

        private TrieNode buildTrie(String[] words){
            
            TrieNode root = new TrieNode();
            for(String word : words){
                TrieNode parent = root;
                int n = word.length();
                for(int i = 0; i < n; i++){
                    char c = word.charAt(i);
                    TrieNode child = parent.childs[c-'a'];
                    if(child == null){
                        child = new TrieNode();
                        parent.childs[c-'a'] = child;
                    }
                    
                    if(i == n-1){
                        child.hasWord = true;
                        child.word = word;
                    }
                    
                    parent = child;
                }
            }
            
            return root;
        }

        class TrieNode{
            String word;
            boolean hasWord;
            TrieNode[] childs = new TrieNode[26];
        }
        
        class UniquePathsWalkOverEmptyCells {
            /**
             * On a 2-dimensional grid, there are 4 types of squares:
             * 1 represents the starting square.  There is exactly one starting square.
             * 2 represents the ending square.  There is exactly one ending square.
             * 0 represents empty squares we can walk over.
             * -1 represents obstacles that we cannot walk over.
             * 
             * Return the number of 4-directional walks from the starting square to the ending 
             * square, that walk over every non-obstacle square exactly once.
             * 
             * Input: [[1,0,0,0],[0,0,0,0],[0,0,2,-1]]
                Output: 2
                Explanation: We have the following two paths: 
                1. (0,0),(0,1),(0,2),(0,3),(1,3),(1,2),(1,1),(1,0),(2,0),(2,1),(2,2)
                2. (0,0),(1,0),(2,0),(2,1),(1,1),(0,1),(0,2),(0,3),(1,3),(1,2),(2,2)
             * 
             */
            int emptyCellCount = 1;
            int count = 0;
            public int uniquePathsIII(int[][] grid) {
                int si = 0, sj = 0;
                for(int i = 0; i < grid.length; i++){
                    for(int j = 0;  j < grid[0].length; j++){
                        if(grid[i][j] == 0){
                            emptyCellCount++;
                        }
                        if(grid[i][j] == 1){
                            si = i; sj = j;
                        }
                    }
                }
                
                uniquePathsIIIWalkDFS(grid, si, sj);
                return count;
            }
            
            public void uniquePathsIIIWalkDFS(int[][] grid, int i, int j){
                // sanity and edge conditions
                if(i < 0 || j < 0 || i > grid.length - 1 || j > grid[0].length - 1 || grid[i][j] < 0){
                    return;
                }
                
                // base case - we reached the end  but have we actually visited all the empty cell?
                // we can track empty cell. As soon as we visit one empty cell we can reduce count
                // as soon as we backtrack we can increase count
                if(grid[i][j] == 2){
                    if(emptyCellCount == 0)
                        count++;
                    return;
                }
                
                // otherwise it is an empty cell
                // traverse all emoty cells
                emptyCellCount--;
                grid[i][j] = -1;
                uniquePathsIIIWalkDFS(grid,i+1, j);
                uniquePathsIIIWalkDFS(grid,i-1, j);
                uniquePathsIIIWalkDFS(grid,i, j+1);
                uniquePathsIIIWalkDFS(grid,i, j-1);
                grid[i][j] = 0; 
                emptyCellCount++;
            }
        }

        public List<int[]> getWalkCandidates(int[][] m, int i, int j, boolean[][] visited) {
            int[][] dirs = new int[][] { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
            List<int[]> cands = new ArrayList<>();
            for (int[] dir : dirs) {
                int k = i + dir[0];
                int l = j + dir[1];

                if (k >= m.length || l >= m[0].length || k < 0 || l < 0) {
                    continue;
                }
                if (!visited[k][l] && canWalk(m, i, j, k, l)) {
                    cands.add(new int[] { k, l });
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
            if (path.size() > maxPath.size()) {
                maxPath = (Stack<Integer>) path.clone();
            }

            List<int[]> cands = getWalkCandidates(m, i, j, visited);
            if (cands.isEmpty()) {
                path.pop();
                return;
            }

            for (int[] cand : cands) {
                walkDFS(m, path, cand[0], cand[1], visited);
            }

            path.pop();
        }

        public List<Integer> walkDFS(int[][] m) {
            for (int i = 0; i < m.length; i++) {
                for (int j = 0; j < m[0].length; j++) {
                    boolean[][] visited = new boolean[m.length][m[0].length];
                    Stack<Integer> path = new Stack<Integer>();
                    walkDFS(m, path, i, j, visited);
                }
            }

            return maxPath;
        }

        // connected components
        class Islands {
            int m;
            int n;
            int[] parent;
            int[] size;
            int count;

            public int gridToSetIndex(int i, int j) {
                return i * n + j;
            }

            public int find(int x) {
                if (parent[x] == x) {
                    return x;
                } else {
                    return find(parent[x]);
                }
            }

            public void union(int x, int y) {
                int rootX = find(x);
                int rootY = find(y);

                if (rootX == rootY)
                    return;
                else {
                    count--;
                }

                if (size[rootX] >= size[rootY]) {
                    size[rootX] += size[rootY];
                    parent[rootY] = rootX;
                } else {
                    size[rootY] += size[rootX];
                    parent[rootX] = rootY;
                }
            }

            public boolean isSameIsland(int x, int y) {
                return find(x) == find(y);
            }

            public int numIslands(char[][] grid) {
                if (grid == null || grid.length == 0 || grid[0].length == 0) {
                    return 0;
                }

                m = grid.length;
                n = grid[0].length;
                parent = new int[m * n];
                size = new int[m * n];
                count = 0;

                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        if (grid[i][j] == '1') {
                            int x = gridToSetIndex(i, j);
                            parent[x] = x;
                            size[x] = 1;
                            count++;
                        }
                    }
                }

                int[][] neighbors = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
                int islands = 0;
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        // if land then test if neighbor grids are also part of same land
                        if (grid[i][j] == '1') {
                            for (int[] nbr : neighbors) {
                                int ni = i + nbr[0];
                                int nj = j + nbr[1];

                                if (ni >= 0 && ni < m && nj >= 0 && nj < n && grid[ni][nj] == '1') {
                                    int x = gridToSetIndex(i, j);
                                    int y = gridToSetIndex(ni, nj);
                                    union(x, y);
                                }
                            }
                        }
                    }
                }

                return count;
            }

            // 1 ms solution
            public int numIslands2(char[][] grid) {
                if (grid == null || grid.length == 0 || grid[0].length == 0) {
                    return 0;
                }

                int m = grid.length;
                int n = grid[0].length;
                int count = 0;

                // for each cell do a connected componenr walk
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        if (grid[i][j] == '1') {
                            // walk to connected component and mark them with same component
                            walkDFS(grid, i, j);
                            count++;
                        }
                    }
                }

                return count;
            }

            private void walkDFS(char[][] grid, int i, int j) {
                if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] == '0') {
                    return;
                }
                grid[i][j] = '0';// mark as visited [same component]
                walkDFS(grid, i + 1, j);
                walkDFS(grid, i, j + 1);
                walkDFS(grid, i - 1, j);
                walkDFS(grid, i, j - 1);
            }

            // 2 ms solution
            public int maxAreaOfIsland(int[][] grid) {
                if (grid == null || grid.length == 0 || grid[0].length == 0) {
                    return 0;
                }

                int m = grid.length;
                int n = grid[0].length;
                int maxArea = 0;

                // for each cell do a connected componenr walk
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        if (grid[i][j] == 1) {
                            // walk to connected component and mark them with same component
                            int area = walkDFSAndComputeArea(grid, i, j);
                            maxArea = Math.max(maxArea, area);
                        }
                    }
                }

                return maxArea;
            }

            private int walkDFSAndComputeArea(int[][] grid, int i, int j) {
                if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] == 0) {
                    return 0;
                }
                grid[i][j] = 0;// mark as visited [same component]
                int area = 1;
                area += walkDFSAndComputeArea(grid, i + 1, j);
                area += walkDFSAndComputeArea(grid, i, j + 1);
                area += walkDFSAndComputeArea(grid, i - 1, j);
                area += walkDFSAndComputeArea(grid, i, j - 1);

                return area;
            }
        }

        class LongestIncreasingPathSolution {
            int[][] lipLens;

            public int longestIncreasingPath(int[][] matrix) {
                if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
                    return 0;
                }

                int m = matrix.length;
                int n = matrix[0].length;
                int maxPathLen = 0;
                lipLens = new int[m][n];

                // for each cell do a connected componenr walk
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        // walk to connected component and mark them with same component
                        int lipLen = walkDFSAndComputeLip(matrix, i, j);
                        maxPathLen = Math.max(maxPathLen, lipLen);
                    }
                }

                return maxPathLen;
            }

            private int walkDFSAndComputeLip(int[][] matrix, int i, int j) {
                if (i < 0 || j < 0 || i >= matrix.length || j >= matrix[0].length) {
                    return 0;
                }
                if (lipLens[i][j] > 0) {
                    return lipLens[i][j];
                }

                int lipLen = 0;
                if (i < matrix.length - 1 && matrix[i + 1][j] > matrix[i][j]) {
                    lipLen = Math.max(lipLen, walkDFSAndComputeLip(matrix, i + 1, j));
                }
                if (j < matrix[0].length - 1 && matrix[i][j + 1] > matrix[i][j]) {
                    lipLen = Math.max(lipLen, walkDFSAndComputeLip(matrix, i, j + 1));
                }
                if (i > 0 && matrix[i - 1][j] > matrix[i][j]) {
                    lipLen = Math.max(lipLen, walkDFSAndComputeLip(matrix, i - 1, j));
                }
                if (j > 0 && matrix[i][j - 1] > matrix[i][j]) {
                    lipLen = Math.max(lipLen, walkDFSAndComputeLip(matrix, i, j - 1));
                }

                lipLens[i][j] = 1 + lipLen;
                return lipLens[i][j];
            }
        }

        class SorroudedRegionsSolution {
            char[][] board;

            public void solve(char[][] board) {
                if (board == null || board.length == 0 || board[0].length == 0) {
                    return;
                }

                this.board = board;
                int m = board.length;
                int n = board[0].length;

                // for each boundary O mark them as non-changeable '-''
                // walk the boundary O's and mark all O's reachable
                // left col
                for (int i = 0; i < m; i++) {
                    walkDFS(i, 0);
                }
                // top row
                for (int j = 0; j < n; j++) {
                    walkDFS(0, j);
                }
                // right col
                for (int i = 0; i < m; i++) {
                    walkDFS(i, n - 1);
                }
                // bottom row
                for (int j = 0; j < n; j++) {
                    walkDFS(m - 1, j);
                }

                // now remaining O's can be flipped to X and the - can be flipped back to O
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        if (board[i][j] == 'O') {
                            board[i][j] = 'X';
                        } else if (board[i][j] == '-') {
                            board[i][j] = 'O';
                        }
                    }
                }
            }

            private void walkDFS(int i, int j) {
                if (i < 0 || j < 0 || i > board.length - 1 || j > board[0].length - 1 || board[i][j] == 'X'
                        || board[i][j] == '-') {
                    return;
                }
                board[i][j] = '-'; // mark as visited [same component]
                walkDFS(i + 1, j);
                walkDFS(i, j + 1);
                walkDFS(i - 1, j);
                walkDFS(i, j - 1);
            }
        }

        class EvaluateOperators {

            public List<String> addOperators(String num, int target) {
                List<String> result = new ArrayList<String>();
                if (num == null || num.length() == 0)
                    return result;

                addOperator(num, 0, "", target, 0, 0, result);
                return result;
            }

            private void addOperator(String num, int curPos, String curPath, int target, long curEval, long prevOpnd,
                    List<String> res) {
                if (curPos == num.length()) {
                    if (curEval == target) {
                        res.add(curPath);
                    }
                    return;
                }

                for (int i = curPos; i < num.length(); i++) {
                    if (i != curPos && num.charAt(curPos) == '0') {
                        break;
                    }

                    long curOprnd = Long.parseLong(num.substring(curPos, i + 1));
                    if (curPos == 0) {
                        addOperator(num, i + 1, curPath + curOprnd, target, curOprnd, curOprnd, res);
                    } else {
                        addOperator(num, i + 1, curPath + "+" + curOprnd, target, curEval + curOprnd, curOprnd, res);
                        addOperator(num, i + 1, curPath + "-" + curOprnd, target, curEval - curOprnd, -curOprnd, res);
                        addOperator(num, i + 1, curPath + "*" + curOprnd, target,
                                curEval - prevOpnd + prevOpnd * curOprnd, prevOpnd * curOprnd, res);
                    }
                }
            }

            public List<String> addOperators2(String num, int target) {
                List<String> result = new ArrayList<String>();
                if (num.length() <= 1) {
                    return result;
                }
                int opnd1 = num.charAt(0) - '0';
                int opnd2 = num.charAt(1) - '0';
                String rem = (num.length() > 2) ? num.substring(2) : "";

                addOperators2(opnd1, opnd2, rem, target, "", result, -1, -1);

                List<String> result2 = new ArrayList<String>();
                for (String res : result) {
                    StringBuffer sb = new StringBuffer();
                    int i = 0;
                    for (i = 0; i < num.length() - 1; i++) {
                        sb.append(num.charAt(i));
                        sb.append(res.charAt(i));
                    }
                    sb.append(num.charAt(i));

                    result2.add(sb.toString());
                }
                return result2;
            }

            private void addOperators2(int opnd1, int opnd2, String remaining, int target, String path,
                    List<String> assignment, int prevopnd1, int prevopnd2) {
                char[] ops = { '+', '*', '-' };

                for (char op : ops) {
                    int newOprnd1 = -1;
                    if (op == '+') {
                        newOprnd1 = opnd1 + opnd2;
                        path += "+";
                    } else if (op == '*') {
                        if (path.isEmpty() || path.endsWith("*")) {
                            newOprnd1 = opnd1 * opnd2;
                        } else if (path.endsWith("+")) {
                            newOprnd1 = prevopnd1 + (prevopnd2 * opnd2);
                        } else if (path.endsWith("-")) {
                            newOprnd1 = prevopnd1 - (prevopnd2 * opnd2);
                        }
                        path += "*";
                    } else if (op == '-') {
                        newOprnd1 = opnd1 - opnd2;
                        path += "-";
                    }

                    if (remaining.isEmpty()) {
                        if (newOprnd1 == target) {
                            assignment.add(path);
                        }
                    } else {
                        int newOprnd2 = remaining.charAt(0) - '0';
                        String newRem = (remaining.length() > 1) ? remaining.substring(1) : "";
                        addOperators2(newOprnd1, newOprnd2, newRem, target, path, assignment, opnd1, opnd2);
                    }

                    path = path.substring(0, path.length() - 1);
                }
            }
        }

        class WordBreak {

            public boolean wordBreak(Set<String> dictionary, String text) {
                // base case
                if (text.isEmpty()) {
                    return true;
                }
                // break the string at i+1 such that prefix text[...i] is in dict and suffix
                // text[i+1...] is breakable
                for (int i = 0; i < text.length(); i++) {
                    if (dictionary.contains(text.substring(0, i + 1)) && wordBreak(dictionary, text.substring(i + 1))) {
                        return true;
                    }
                }

                return false;
            }

            public boolean wordBreak(Set<String> dictionary, String text, ArrayList<String> result) {
                // base case
                if (text.isEmpty()) {
                    return true;
                }
                // break the string at i+1 such that prefix text[...i] is in dict and suffix
                // text[i+1...] is breakable
                for (int i = 0; i < text.length(); i++) {
                    if (dictionary.contains(text.substring(0, i + 1))
                            && wordBreak(dictionary, text.substring(i + 1), result)) {
                        result.add(0, text.substring(0, i + 1));
                        return true;
                    }
                }

                return false;
            }
            
            public boolean wordBreakDP(Set<String> dictionary, String text) {
                int n = text.length();
                if(n == 0){
                    return true;
                }
                
                //dp[i] = true if there is a solution in prefix text[0..i]
                boolean[] dp = new boolean[n+1];  
                dp[0] = true;// base case empty string can always be brekable 
                
                //try all possible prefixes
                for(int i = 1; i<= n; i++){
                    // break into prefix and suffix
                    // for s[0..i] prefixs we break into s[0..j] prefix and s[j..i] suffix 
                    for(int j = 0; j < i; j++){
                        // check if prefix has solution (subproblem) and suffix is in dictionary
                        if(dp[j] && dictionary.contains(text.substring(j, i))){
                            dp[i] = true;
                            break;
                        }
                    }
                }
                
                return dp[n];
            }

            /**
             *  Input:
                s = "catsanddog"
                wordDict = ["cat", "cats", "and", "sand", "dog"]
                Output:
                [
                  "cats and dog",
                  "cat sand dog"
                ]

             * @param dictionary
             * @param text
             * @param dpMap
             * @return
             */
            public List<String> wordBreakAll(Set<String> dictionary, String text, Map<String, List<String>> dpMap) {
                // if already computed the current substring text then return from map
                if (dpMap.containsKey(text)) {
                    return dpMap.get(text);
                }
                List<String> result = new ArrayList<String>();

                // if the whole word is in the dictionary then we add this to final result
                if (dictionary.contains(text)) {
                    result.add(text);
                }

                // try each prefix and extend
                for (int i = 0; i < text.length(); i++) {
                    // take a prefix and recursively chck if the remaining (suffix) can be broken
                    String prefix = text.substring(0, i + 1);
                    if (dictionary.contains(prefix)) {
                        // extend
                        String suffix = text.substring(i + 1);
                        List<String> subRes = wordBreakAll(dictionary, suffix, dpMap);
                        // for each result list from the suffix make the final answer by
                        // appending prefix to the front of each answer
                        for (String word : subRes) {
                            result.add(prefix + " " + word);
                        }
                    }
                }

                // cache the result for later use
                dpMap.put(text, result);
                return result;
            }
            
            public int wordBreakCountAll(Set<String> dictionary, String text, Map<String, Integer> dpMap) {
                // if already computed the current substring text then return from map
                if (dpMap.containsKey(text)) {
                    return dpMap.get(text);
                }
                int count = 0;

                // if the whole word is in the dictionary then we add this to final result
                if (dictionary.contains(text)) {
                    count++;
                }

                // try each prefix and extend
                for (int i = 0; i < text.length(); i++) {
                    // take a prefix and recursively chck if the remaining (suffix) can be broken
                    String prefix = text.substring(0, i + 1);
                    if (dictionary.contains(prefix)) {
                        // extend
                        String suffix = text.substring(i + 1);
                        count += wordBreakCountAll(dictionary, suffix, dpMap);
                    }
                }

                // cache the result for later use
                dpMap.put(text, count);
                return count;
            }
            
            /**
             * Given a list of words (without duplicates), please write a program that returns 
             * all concatenated words in the given list of words.
             * 
             * A concatenated word is defined as a string that is comprised entirely of at least 
             * two shorter words in the given array.
             * 
             * Input: 
             * ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]
             * Output: ["catsdogcats","dogcatsdog","ratcatdogcat"]

             * @param words
             * @return
             */
            public List<String> findAllConcatenatedWordsInADict(String[] words) {
                if(words.length == 0){
                    return Collections.emptyList();
                }
                
                Set<String> dictionary = new HashSet<>(Arrays.asList(words));
                List<String> res = new ArrayList<>();
                // instead of concatenate, for each word test if they are breakable 
                // remove the world itself from the dictionary so that the solution is comprised 
                // entirely of at least two shorter words in the given array.
                for(String w : words){
                    if(!w.isEmpty()){
                        dictionary.remove(w);
                        if(wordBreakDP(dictionary, w)){
                            res.add(w);
                        }
                        dictionary.add(w);
                    }
                }
                
                return res;
            }
            
            /**
             * A message containing letters from A-Z is being encoded to numbers using the following mapping:

                    'A' -> 1
                    'B' -> 2
                    ...
                    'Z' -> 26

               Given a non-empty string containing only digits, determine the total number of ways to decode it.
               Input: s = "12"
                Output: 2
                Explanation: It could be decoded as "AB" (1 2) or "L" (12).
                
             * @param s
             * @return
             */
            public int numDecodings(String s) {
                Set<String> dictionary = new HashSet<>();
                for(int n = 1; n <= 26; n++){
                    dictionary.add(n+"");
                }
                
                return wordBreakCountAll(dictionary, s, new HashMap<>());
            }
            
            /**
             * Given a string s, partition s such that every substring of the partition is a palindrome.
             * Return all possible palindrome partitioning of s.

                Example:
                
                Input: "aab"
                Output:
                [
                  ["aa","b"],
                  ["a","a","b"]
                ]

             * @param s
             * @return
             */
            public List<List<String>> partitionIntoPalindrom(String s) {
                if(s.isEmpty()){
                    List<List<String>> empty = new ArrayList<>();
                    empty.add(Collections.emptyList());
                    return empty;
                }
                
                return partitionPalinromAll(s, new HashMap<>());
            }
            
            // use word break to break the word into palindromes
            public List<List<String>> partitionPalinromAll(String text, Map<String, List<List<String>>> dpMap) {
                // if already computed the current substring text then return from map
                if (dpMap.containsKey(text)) {
                    return dpMap.get(text);
                }
                List<List<String>> result = new ArrayList<>();

                // if the whole word is in the dictionary then we add this to final result
                if (isPalindrom(text)) {
                    result.add(Arrays.asList(new String[]{text}));
                }

                // try each prefix and extend
                for (int i = 0; i < text.length(); i++) {
                    // take a prefix and recursively chck if the remaining (suffix) can be broken
                    String prefix = text.substring(0, i + 1);
                    if (isPalindrom(prefix)) {
                        // extend
                        String suffix = text.substring(i + 1);
                        List<List<String>> subRes = partitionPalinromAll(suffix, dpMap);
                        // for each result list from the suffix make the final answer by
                        // appending prefix to the front of each answer
                        for (List<String> sub : subRes) {
                            List<String> s =  new ArrayList<>();
                            s.add(prefix);
                            s.addAll(sub);
                            result.add(s);
                        }
                    }
                }

                // cache the result for later use
                dpMap.put(text, result);
                return result;
            }
            
            private boolean isPalindrom(String w){
                if(w.length() == 0){
                    return false;
                }
                if(w.length() == 1){
                    return true;
                }
                int i = 0, j = w.length()-1;
                while(i < j){
                    if(w.charAt(i) != w.charAt(j)){
                        return false;
                    }
                    i++;j--;
                }
                
                return true;
            }
        }

        class Factors {

            public void printFactors(int number) {
                printFactors("", number, number);
            }

            public void printFactors(String expression, int dividend, int previous) {
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
        }
    }

    class LRUCache {

        class DLLNode{
            public int key;
            public int value;
            public DLLNode next;
            public DLLNode prev;
            
            public DLLNode(int key, int value){
                this.key = key;
                this.value = value;
            }
        }
        
        private Map<Integer, DLLNode> cache;
        private DLLNode dummyHead;
        private DLLNode dummyTail;
        int capacity;
        
        public LRUCache(int capacity) {
            this.capacity = capacity;
            cache = new LinkedHashMap<>(capacity);
            // setup empty DLL
            dummyHead = new DLLNode(-1, -1);
            dummyTail = new DLLNode(-1, -1);
            dummyHead.next = dummyTail;
            dummyTail.prev = dummyHead;
        }
        
        public int get(int key) {
            if(cache.containsKey(key)){
                // move the node up to the front
                DLLNode node = cache.get(key);
                delete(node);
                addLast(node);
                return node.value;
            }
            
            return -1;
        }
        
        public void put(int key, int value) {
            if(cache.containsKey(key)){
                cache.get(key).value = value;
            }
            else{
                if(cache.size() == this.capacity){
                    DLLNode node = deleteFirst();
                    cache.remove(node.key);
                }
                
                DLLNode node = new DLLNode(key, value);
                addLast(node);
                cache.put(key, node);
            }
        }
        
        private boolean addLast(DLLNode node){
            DLLNode prev = dummyTail.prev;
            dummyTail.prev = node;
            node.next = dummyTail;
            node.prev = prev;
            prev.next = node;
            
            return true;
        }
        
        private DLLNode deleteFirst() {
            DLLNode next = dummyHead.next;
            dummyHead.next = next.next;
            next.next.prev = dummyHead;
            
            return next;
        }
        
        private boolean delete(DLLNode node) {
            if(cache.size() == 0){
                return false;
            }
            
            DLLNode prev = node.prev;
            DLLNode next = node.next;
            
            prev.next = node.next;
            next.prev = prev;
            
            return true;
        }
    }

    
    class Substrings {

        public int strStr(String s, String t) {
            if (t.isEmpty())
                return 0; // edge case: "",""=>0 "a",""=>0
            for (int i = 0; i <= s.length() - t.length(); i++) {
                for (int j = 0; j < t.length() && s.charAt(i + j) == t.charAt(j); j++)
                    if (j == t.length() - 1)
                        return i;
            }
            return -1;
        }

        private int match(int[] text, int[] patt) {
            int count = 0;
            for (int i = 0; i < patt.length; i++) {
                if (patt[i] != 0 && text[i] != 0 && text[i] >= patt[i]) {
                    count++;
                }
            }

            return count;
        }
        
        /**
         * Given two strings s1 and s2, write a function to return true if s2 contains the permutation 
         * of s1. In other words, one of the first string's permutations is the substring of the second string.
         * 
         * Input: s1 = "ab" s2 = "eidbaooo"
         * Output: True
         * Explanation: s2 contains one permutation of s1 ("ba").

         * @param s1
         * @param s2
         * @return
         */
        public boolean checkInclusion(String s1, String s2) {
            int[] hist = new int[26];
            if(s1.length() > s2.length()){
               return false;
            }
            
            for(int i = 0; i < s1.length(); i++){
                // make a sliding window of freq of legth of s1
                hist[s1.charAt(i) - 'a']++;
                hist[s2.charAt(i) - 'a']--;
            }
            if(matches(hist)) return true;
            
            // now slide the window
            for(int j = s1.length(); j < s2.length(); j++){
                hist[s2.charAt(j-s1.length()) - 'a']++;
                hist[s2.charAt(j) - 'a']--;
                if(matches(hist)) return true;
            }
            
            return false;
        }

        private boolean matches(int[] count) {
            for (int i = 0; i < 26; i++) {
                if (count[i] != 0) return false;
            }
            return true;
        }


        /**
         * Given two strings s and t, return the minimum window in s which will contain all the characters in t.
         * If there is no such window in s that covers all characters in t, return the empty string "".
         * Note that If there is such a window, it is guaranteed that there will always be only one unique minimum window in s.
         * 
         * Input: s = "ADOBECODEBANC", t = "ABC"
         * Output: "BANC"
         * 
         * Input: s = "a", t = "a"
         * Output: "a"
         * 
         * @param s
         * @param t
         * @return
         */
        public String minLenSuperSubString1(String s, String t) {
            if (t.length() > s.length()) {
                return "";
            }
            if (s.equals(t)) {
                return s;
            }

            // keep a map for char frequency of the pattern
            int[] hist = new int[256];
            for (int i = 0; i < t.length(); i++) {
                hist[t.charAt(i) - 'A']++;
            }
            // slide the window till we match all the chars then try to shrink
            int i = 0, j = 0;
            int bestStart = 0;
            int minLen = Integer.MAX_VALUE;
            int targetCount = t.length();
            while (j < s.length()) {
                // to expand the window decrease counter only if the matching char found
                // (counter positive )
                if (hist[s.charAt(j) - 'A'] > 0) {
                    targetCount--;
                }
                // expand the window from end
                hist[s.charAt(j++) - 'A']--;

                // shrink the window from front
                while (targetCount == 0) {
                    if (minLen > j - i) {
                        minLen = j - i;
                        bestStart = i;
                    }

                    // to shrink the window increase counter only if it was a matching character
                    // (counter not negative)
                    if (hist[s.charAt(i) - 'A'] >= 0) {
                        targetCount++;
                    }
                    // shrink the window from the front
                    hist[s.charAt(i++) - 'A']++;
                }
            }

            return ((minLen != Integer.MAX_VALUE) ? s.substring(bestStart, bestStart + minLen) : "");
        }
        
        public String minLenSuperSubString2(String s, String t) {
            if (t.length() > s.length()) {
                return "";
            }
            if (s.equals(t)) {
                return s;
            }

            int[] histS = new int[256];
            Arrays.fill(histS, 0);
            int[] histT = new int[256];
            Arrays.fill(histT, 0);

            for (char c : t.toCharArray()) {
                histT[c - 'A']++;
            }

            int start = 0, bestStart = 0, len = 0, minLen = Integer.MAX_VALUE;
            int targetMatchCount = match(histT, histT);

            int j = start;
            while (start < s.length()) {
                int matchCount = match(histS, histT);

                // increase the window forward as long as we don't match all the
                // chars in t at least once
                while (j < s.length() && matchCount < targetMatchCount) {
                    histS[s.charAt(j) - 'A']++;
                    matchCount = match(histS, histT);
                    j++;
                }

                // no solution from this start position
                if (matchCount < targetMatchCount) {
                    break;
                }
                // we found a substring
                else {
                    len = j - start;
                    if (len < minLen) {
                        minLen = len;
                        bestStart = start;
                    }
                }

                // try to shrink the window
                histS[s.charAt(start) - 'A']--;
                start++;
            }

            if (bestStart + minLen <= s.length())
                return s.substring(bestStart, bestStart + minLen);

            return "";
        }
        
        // 2 ms 39 MB
        public int lengthOfLongestSubstring1(String s) {
            if (s.length() == 0) {
                return 0;
            }

            // slide the window till we match all the chars then try to shrink
            int i = 0, j = 0;
            int maxLen = 0;
            int targetCount = 0;
            int[] hist = new int[128];
            while (j < s.length()) {
                // to expand the window increase counter only if the matching char found
                // (counter positive )
                if (hist[s.charAt(j)] > 0) {
                    targetCount++;
                }
                // expand the window from end
                hist[s.charAt(j++) - 'A']++;

                // shrink the window from front
                while (targetCount > 0) {
                    // to shrink the window decrease counter only if it was a matching character
                    // (counter positive)
                    if (hist[s.charAt(i)] > 1) {
                        targetCount--;
                    }
                    // shrink the window from the front
                    hist[s.charAt(i++)]--;
                }

                maxLen = Math.max(maxLen, j - i);
            }

            return maxLen;
        }

        // 2 ms 36 MB
        public int lengthOfLongestNonrepeatedSubstring(String s) {
            if (s == null || s.isEmpty()) {
                return 0;
            }

            int lastIndices[] = new int[128];
            for (int i = 0; i < 128; i++) {
                lastIndices[i] = -1;
            }

            int maxLen = 0;
            int curLen = 0;
            int start = 0;
            int bestStart = 0;
            for (int i = 0; i < s.length(); i++) {
                char cur = s.charAt(i);
                if (lastIndices[cur] < start) {
                    lastIndices[cur] = i;
                    curLen++;
                } else {
                    int lastIndex = lastIndices[cur];
                    start = lastIndex + 1;
                    curLen = i - start + 1;
                    lastIndices[cur] = i;
                }

                if (curLen > maxLen) {
                    maxLen = curLen;
                    bestStart = start;
                }
            }

            return maxLen;
        }

        public int lengthOfLongestSubstringAtMostTwoDistinct(String s) {
            if (s.length() == 0) {
                return 0;
            }

            // slide the window till we match all the chars then try to shrink
            int i = 0, j = 0;
            int maxLen = 0;
            int targetCount = 0;
            int[] hist = new int[256];
            while (j < s.length()) {
                // to expand the window increase counter if non (zero) or some (> 0) character
                // matches
                if (hist[s.charAt(j)] == 0) {
                    targetCount++;
                }
                // expand the window from end
                hist[s.charAt(j++) - 'A']++;

                // shrink the window from front
                while (targetCount > 2) {
                    // to shrink the window decrease counter only if it was a matching character
                    // (counter positive)
                    if (hist[s.charAt(i)] > 1) {
                        targetCount--;
                    }
                    // shrink the window from the front
                    hist[s.charAt(i++)]--;
                }

                maxLen = Math.max(maxLen, j - i);
            }

            return maxLen;
        }

        public int lengthOfLongestSubstringAtMostKDistinct(String s, int k) {
            if (s.length() == 0 || k == 0) {
                return 0;
            }

            // slide the window till we match all the chars then try to shrink
            int i = 0, j = 0;
            int maxLen = 0;
            int targetCount = 0;
            int[] hist = new int[256];
            while (j < s.length()) {
                // to expand the window increase counter if non (zero) or some (> 0) character
                // matches
                if (hist[s.charAt(j)] == 0) {
                    targetCount++;
                }
                // expand the window from end
                hist[s.charAt(j++) - 'A']++;

                // shrink the window from front
                while (targetCount > k) {
                    // to shrink the window decrease counter only if it was a matching character
                    // (counter positive)
                    if (hist[s.charAt(i)] > k - 1) {
                        targetCount--;
                    }
                    // shrink the window from the front
                    hist[s.charAt(i++)]--;
                }

                maxLen = Math.max(maxLen, j - i);
            }

            return maxLen;
        }

        // smallest lexicographic string after removing duplicates
        public String lexicoSmallNoDuplicates(String str) {
            int[] hist = new int[256];
            StringBuilder out = new StringBuilder();

            // compute character count histogram
            for (int i = 0; i < str.length(); i++) {
                hist[str.charAt(i) - '0']++;
            }

            // scan left to right and remove current if and only if -
            // count for cur character is > 1 and value of character is lexicographically
            // greater than next character. Otherwise we take the character (if not already
            // taken early)
            for (int i = 0; i < str.length() - 2; i++) {
                int cur = str.charAt(i) - '0';
                int next = str.charAt(i + 1) - '0';
                if (cur > next && hist[cur] > 1) {
                    hist[cur]--;
                } else if (hist[cur] != 0) {
                    out.append(str.charAt(i));
                    hist[cur] = 0;
                }
            }

            if (hist[str.charAt(str.length() - 1) - '0'] != 0) {
                out.append(str.charAt(str.length() - 1));
            }

            return out.toString();
        }

        public int findMaxRepeatedSubStrLength(int[] A, int[] B) {
            if (A == null || B == null) {
                return 0;
            }

            int[][] dp = new int[A.length + 1][B.length + 1];
            int max = 0;

            for (int i = 1; i <= A.length; i++) {
                for (int j = 1; j <= B.length; j++) {
                    if (A[i - 1] == B[j - 1]) {
                        dp[i][j] = 1 + dp[i - 1][j - 1];
                        max = Math.max(max, dp[i][j]);
                    }
                }
            }

            return max;
        }
    }

    class BinarySearch {

        public int[] searchRange(int[] nums, int target) {
            int res[] = new int[] { -1, -1 };
            if (nums == null || nums.length == 0) {
                return res;
            }
            if (nums.length == 1) {
                if (nums[0] == target) {
                    return new int[] { 0, 0 };
                } else
                    return res;
            }

            // find the left most target
            int l = 0;
            int h = nums.length - 1;
            int mid = (h + l) / 2;
            while (l < h) {
                mid = (h + l) / 2;
                // if we find target - keep continue to search in the left
                if (nums[mid] == target) {
                    h = mid;
                }
                // if mid is higher then search in left part
                else if (nums[mid] > target) {
                    h = mid-1;
                } else {
                    l = mid + 1;
                }
            }

            if (nums[l] == target) {
                res[0] = l;
            } else {
                return res;
            }

            // find the right most target
            h = nums.length - 1;
            while (l < h) {
                mid = (h + l) / 2 + 1; // make mid to the right

                // if we find target - keep continue to search in the right
                if (nums[mid] == target) {
                    l = mid;
                }
                // if mid is higher then search in left part
                else if (nums[mid] > target) {
                    h = mid - 1;
                } else {
                    l = mid+1;
                }
            }

            res[1] = h;

            return res;
        }

        public int findPeakElement(int[] nums) {
            int l = 0;
            int h = nums.length - 1;
            int mid = 0;

            while (l < h) {
                mid = l + (h - l) / 2;
                if (nums[mid] < nums[mid + 1]) {
                    l = mid + 1;
                } else {
                    h = mid;
                }
            }

            return l;
        }

        public int searchRotated(int[] nums, int target) {
            int n = nums.length;
            int l = 0;
            int h = n - 1;
            // find the index of the smallest value using binary search.
            // Loop will terminate since mid < hi, and lo or hi will shrink by at least 1.
            // Proof by contradiction that mid < hi: if mid==hi, then lo==hi and loop would
            // have been terminated.
            while (l < h) {
                int mid = (l + h) / 2;

                // rotated part
                if (nums[mid] > nums[h]) {
                    l = mid + 1;
                }
                // sorted part
                else {
                    h = mid;
                }
            }

            // lo==hi is the index of the smallest value (pivot) and also the number of
            // places rotated.
            int pivot = l;
            // The usual binary search and accounting for rotation.
            l = 0;
            h = n - 1;
            while (l <= h) {
                int mid = (l + h) / 2;
                int actualmid = (mid + pivot) % n;

                if (nums[actualmid] == target) {
                    return actualmid;
                } else if (nums[actualmid] < target) {
                    l = mid + 1;
                } else {
                    h = mid - 1;
                }
            }

            return -1;
        }

        public int searchRotationPosition(int a[]) {
            int n = a.length;
            int l = 0;
            int h = a.length - 1;
            int mid;

            while (l < h) {
                mid = (l + h) / 2;

                if (mid > 0 && mid < n - 1 && a[mid] < a[mid - 1] && a[mid] <= a[mid + 1]) {
                    return mid;
                } else if (a[mid] >= a[h]) {
                    l = mid + 1;
                } else {
                    h = mid - 1;
                }
            }

            return -1;
        }

        public int findRotationPositin(final int[] a) {
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

        // max value less than equal to key
        public int floor(int[] a, int key) {
            int l = 0;
            int h = a.length - 1;

            if (key < a[l]) {
                return -1;
            }

            while (l < h) {
                int mid = l + (h - l) / 2;

                if (a[mid] == key)
                    return mid;
                else if (a[mid] < key) {
                    if (mid < a.length - 1 && a[mid + 1] >= key) {
                        return mid;
                    }
                    l = mid + 1;
                } else {
                    h = mid - 1;
                }
            }

            return l;
        }

        // min value greater than equal to key
        public int ceil(int[] a, int key) {
            int l = 0;
            int h = a.length - 1;

            if (key > a[h]) {
                return -1;
            }

            while (l < h) {
                int mid = l + (h - l) / 2;

                if (a[mid] == key)
                    return mid;
                else if (a[mid] < key) {
                    l = mid + 1;
                } else {
                    if (mid > 0 && a[mid - 1] < key) {
                        return mid;
                    }
                    h = mid - 1;
                }
            }

            return l;
        }

        public int binarySearchClosest(int a[], int l, int h, int key) {
            if (a.length == 0 || l < 0 || h > a.length - 1) {
                return -1;
            }

            while (true) {
                if (h < l) {
                    return l;
                }
                int mid = l + (h - l) / 2;

                if (a[mid] == key) {
                    return mid;
                }
                if (a[mid] > key) {
                    h = mid - 1;
                } else {
                    l = mid + 1;
                }
            }
        }
    }

    class BinarySearchTree {
        
        public TreeNode sortedArrayToBST(int[] nums) {
            return sortedArrayToBSTHelper(nums, 0, nums.length-1);
        }
        
        private TreeNode sortedArrayToBSTHelper(int nums[], int i, int j){
            if(i > j){
                return null;
            }
            
            int mid = i + (j-i)/2;
            TreeNode root = new TreeNode(nums[mid]);
            root.left = sortedArrayToBSTHelper(nums, i, mid-1);
            root.right = sortedArrayToBSTHelper(nums, mid+1, j);
            return root;
        }

        public boolean isValidBST1(TreeNode root) {
            if (root == null) {
                return true;
            }

            boolean leftValid = true;
            if (root.left != null) {
                leftValid = (root.val > root.left.val);
            }
            boolean rightValid = true;
            if (root.right != null) {
                rightValid = (root.val < root.right.val);
            }

            return isValidBST(root.left) && isValidBST(root.right) && leftValid && rightValid;
        }

        public boolean isValidBST(TreeNode root) {
            return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
        }

        public boolean isValidBST(TreeNode root, long minVal, long maxVal) {
            if (root == null)
                return true;
            if (root.val >= maxVal || root.val <= minVal)
                return false;
            return isValidBST(root.left, minVal, root.val) && isValidBST(root.right, root.val, maxVal);
        }
        
        public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
            if (root == null) {
                return null;
            }

            while (root != null) {
                if (root.val > p.val && root.val > q.val) {
                    root = root.left;
                } else if (root.val < p.val && root.val < q.val) {
                    root = root.right;
                } else {
                    return root;
                }
            }
            return null;
        }

        public boolean isBSTPostOrder(int[] a, int p, int q) {
            int n = q - p + 1;
            ;
            // base case always true for 1 element
            if (n < 2) {
                return true;
            }

            // partition into left subtree a[p..right-1] and right subtree a[right..q-1]
            int right = p;
            while (a[right] < a[q])
                right++;

            // check validity of right subtree
            int i = right;
            while (i < q && a[i] > a[q])
                i++;
            if (i < q) {
                return false;
            }

            return isBSTPostOrder(a, p, right - 1) && isBSTPostOrder(a, right, q - 1);
        }

        public int numOfUniqueBST1(int len) {
            if (len <= 1) {
                return 1;
            } else {
                int count = 0;
                for (int i = 1; i <= len; i++) {
                    count += numOfUniqueBST1(i - 1) * numOfUniqueBST1(len - i);
                }
                return count;
            }
        }

        public int numOfUniqueBST1(int m, int n) {
            int len = n - m + 1;
            return numOfUniqueBST1(len);
        }

        public int numOfUniqueBSTDP(int n, int[] counts) {
            int count = 0;
            if (n < 0) {
                return 0;
            }
            if (n <= 1) {
                return 1;
            }

            // count possible trees with each element as root
            for (int i = 1; i <= n; i++) {
                // compute if not in DP
                if (counts[i] == -1) {
                    counts[i] = numOfUniqueBSTDP(i - 1, counts);
                }
                if (counts[n - i] == -1) {
                    counts[n - i] = numOfUniqueBSTDP(n - i, counts);
                }

                count += counts[i - 1] + counts[n - i];
            }

            counts[n] = count;
            return count;
        }

        public int numOfUniqueBSTDP(int m, int n) {
            int len = n - m + 1;
            int[] counts = new int[n + 1];

            // mark each cell not computed
            for (int i = 0; i <= n; i++) {
                counts[i] = -1;
            }

            return numOfUniqueBSTDP(len, counts);
        }

        public class DLLListToBSTInplace {
            ListNode head;

            private ListNode convert(int start, int end) {
                if (start > end) {
                    return null;
                }

                int mid = start + (end - start) / 2;
                ListNode left = convert(start, mid - 1);
                ListNode root = head;
                head = head.next;
                root.prev = left;
                root.next = convert(mid + 1, end);

                return root;
            }

            public ListNode convert() {
                int n = 0;
                ListNode h = head;
                while (h != null) {
                    n++;
                    h = h.next;
                }

                return convert(0, n - 1);
            }

            public DLLListToBSTInplace(ListNode head) {
                this.head = head;
            }
        }
        
        public int kthSmallestInBST(TreeNode root, int k) {
            if(root == null){
                return -1;
            }
            
            TreeNode cur = root;
            TreeNode pre = null;
            
            while(cur != null){
                // if no left subtree then visit right subtree
                if(cur.left == null){
                    if(--k == 0){
                        return cur.val;
                    }
                    cur = cur.right;
                }
                else{
                    // we have left subtree. We need to visit this subtree. The left subtree visit needs
                    // to come back to the cur node. In order to the predecessor (max node in left subyree)
                    // to rerach the cur node we can thread it's unused right pointer to point to cur
                    // when traverse comes back to cur node pre.right is pointing to cur, so stop the loop
                    pre = cur.left;
                    while(pre.right != null && pre.right != cur){
                        pre = pre.right;
                    }
                    
                    // pre.right is null here before setting up the threaded pointer
                    // set up the back pointer and then go visit he left subtree
                    if(pre.right == null){
                        pre.right = cur;
                        cur = cur.left;
                    }
                    // otherwise the traverdal has came back to the root node (successor) from the 
                    // left subtree. That means we now can visit the root and then continue to right subtree
                    else{
                        cur = pre.right;
                        if(--k == 0){
                            return cur.val;
                        }
                        // first unlink the thread to avoid loop
                        pre.right = null;
                        cur = cur.right;
                    }
                }
            }
            
            return -1;
        }
    }

    class ThreeSum {
        public List<List<Integer>> threeSum(int[] nums) {
            List<List<Integer>> result = new LinkedList<>();
            Arrays.sort(nums);

            for (int k = 0; k < nums.length - 2; k++) {
                if (k == 0 || (k > 0 && nums[k] != nums[k - 1])) {
                    int sumKey = 0 - nums[k];
                    int i = k + 1, j = nums.length - 1;
                    while (i < j) {
                        if ((nums[i] + nums[j]) > sumKey) {
                            j--;
                        } else if ((nums[i] + nums[j]) < sumKey) {
                            i++;
                        } else {
                            result.add(Arrays.asList(nums[k], nums[i], nums[j]));
                            while (i < j && nums[i] == nums[i + 1])
                                i++;
                            while (i < j && nums[j] == nums[j - 1])
                                j--;
                            i++;
                            j--;
                        }
                    }
                }
            }

            return result;
        }

        public int threeSumClosest(int[] nums, int target) {
            if (nums.length < 3) {
                return -1;
            }

            int result = nums[0] + nums[1] + nums[nums.length - 1];
            if (nums.length == 3) {
                return result;
            }

            Arrays.sort(nums);

            for (int i = 0; i < nums.length - 2; i++) {
                if (i == 0 || (i > 0 && nums[i] != nums[i - 1])) {
                    int j = i + 1, k = nums.length - 1;

                    while (j < k) {
                        int sum = nums[i] + nums[j] + nums[k];
                        if (sum == target)
                            return sum;

                        if (sum < target) {
                            j++;
                        } else {
                            k--;
                        }

                        result = Math.abs(sum - target) < Math.abs(result - target) ? sum : result;
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
            if (a.length == 0) {
                return -1;
            }
            if (a.length == 1) {
                return 0;
            } else if (a.length == 2) {
                return closer(a[0], a[1], key) ? 0 : 1;
            } else if (a[0] >= key) {
                return 0;
            } else if (a[a.length - 1] <= key) {
                return a.length - 1;
            }

            int l = 0;
            int h = a.length - 1;
            int mid = 0;

            while (l < h) {
                mid = l + (h - l) / 2;

                if (a[mid] == key) {
                    return mid;
                }
                // mid is already higher; then either mid is closest or it is in a[l..mid-1]
                if (a[mid] > key) {
                    if (closer(a[mid], a[mid - 1], key)) {
                        return mid;
                    }
                    h = mid - 1;
                }
                // mid is already lesser; then either mid is closest or it is in a[mid+1..h]
                else {
                    if (closer(a[mid], a[mid + 1], key)) {
                        return mid;
                    }
                    l = mid + 1;
                }

            }

            return -1;
        }

        public int minDiffElement(int a[], int l, int h, int key) {
            if (a.length == 0 || l < 0 || h > a.length - 1) {
                return -1;
            }
            if (l == h) {
                return l;
            } else if (h == l + 1) {
                return closer(a[l], a[h], key) ? l : h;
            } else if (a[l] >= key) {
                return l;
            } else if (a[h] <= key) {
                return h;
            }

            while (l < h) {
                int mid = l + (h - l) / 2;

                if (a[mid] == key) {
                    return mid;
                }
                // mid is already higher; then either mid is closest or it is in a[l..mid-1]
                if (a[mid] > key) {
                    if (closer(a[mid], a[mid - 1], key)) {
                        return mid;
                    }
                    h = mid - 1;
                }
                // mid is already lesser; then either mid is closest or it is in a[mid+1..h]
                else {
                    if (closer(a[mid], a[mid + 1], key)) {
                        return mid;
                    }
                    l = mid + 1;
                }

            }

            return -1;
        }

//        public List<List<Integer>> threeSum2(int[] a) {
//            List<List<Integer>> res = new ArrayList<>();
//            
//            if(a.length < 3) {
//                return res;
//            }
//            Arrays.sort(a);
//            
//            int l = 0;
//            int h = a.length-1;
//            while(l < h && (h-l+1) >=3){
//                int third = 0 - (a[l]+a[h]);
//                int closestKey = minDiffElement(a, l+1, h-1, third);
//                int sum = a[l]+a[h]+a[closestKey];
//                
//                if(sum == 0){
//                    res.add(Arrays.asList(a[l], a[closestKey], a[h]));
//                    l++;
//                    h--;
//                    // handle duplicate cases
//                    while(l < h && (a[l] == a[l-1]) && (a[h] == a[h+1])) {
//                        l++;
//                        h--;
//                    }
//                }
//                else if(sum < 0){
//                    l++;
//                }
//                else{
//                    h--;
//                }
//            }
//            
//            return res;
//        }

        // accepted 14 ms
        // O(nlgn) < Oder < O(n^2)
        public List<List<Integer>> threeSum3(int[] nums) {
            List<List<Integer>> result = new LinkedList<>();
            Arrays.sort(nums);
            BinarySearch bs = new BinarySearch();

            for (int i = 0; i < nums.length - 2; i++) {
                if (i == 0 || (i > 0 && nums[i] != nums[i - 1])) {
                    // unoptiomized
                    int j = i + 1, k = nums.length - 1;
                    // Each time we increment i, we pick the next k by selecting the last number in
                    // the
                    // list, which is also the maximum number. Now, once i gets closer and closer to
                    // zero (in the sorted list),
                    // this k will be less appropriate and we will end up doing a linear search from
                    // k
                    // towards i to get a smaller positive number that will give us a zero sum.
                    // The optimal k can be found more efficiently with a binary search, which makes
                    // the overall algorithm more efficient again.
                    // A similar argument applies to j. If we consider the maximum number, it may
                    // make less sense
                    // to make j become i+1 for its first candidate. Namely if i+j+k==0, then j
                    // should be -i + -k.
                    j = bs.binarySearchClosest(nums, i + 1, nums.length - 2, -(nums[i] + nums[nums.length - 1]));
                    k = bs.binarySearchClosest(nums, j + 1, nums.length - 1, -(nums[i] + nums[j]));

                    while (j < k && k < nums.length) {
                        int sum = nums[i] + nums[j] + nums[k];

                        if (sum < 0) {
                            j++;
                        } else if (sum > 0) {
                            k--;
                        } else {
                            result.add(Arrays.asList(nums[i], nums[j], nums[k]));
                            while (j < k && nums[j] == nums[j + 1])
                                j++;
                            while (j < k && nums[k] == nums[k - 1])
                                k--;
                            j++;
                            k--;
                        }
                    }
                }
            }

            return result;
        }

        public List<List<Integer>> fourSum(int[] nums, int target) {
            return kSum(nums, target, 4);
        }

        public List<List<Integer>> kSum(int[] nums, int targetSum, int k) {
            if (nums == null || nums.length < k || k < 2) {
                return new ArrayList<>();
            }
            Arrays.sort(nums);
            return kSumHelperOnSorted(nums, targetSum, k, 0);
        }

        public List<List<Integer>> kSumHelperOnSorted(int[] nums, int targetSum, int k, int start) {
            int n = nums.length;
            ArrayList<List<Integer>> result = new ArrayList<List<Integer>>();

            if (start >= n || n - start + 1 < k || k < 2 || targetSum < nums[start] * k
                    || targetSum > nums[n - 1] * k) {
                return result;
            }

            if (k == 2) {
                return twoSumOnSorted(nums, start, n - 1, targetSum);
            } else {

                for (int i = start; i < n - k + 1; i++) {
                    // skip duplicates
                    if (i == start || (i > start && nums[i - 1] != nums[i])) {
                        // for each nums[i] recursively find the next k-1 numbers that along with
                        // nums[i] can make the total sum to be targetSum
                        List<List<Integer>> subProblemResult = kSumHelperOnSorted(nums, targetSum - nums[i], k - 1,
                                i + 1);

                        // add nums[i] to the (k-1) subproblem result list to make it k sum result
                        for (List<Integer> res : subProblemResult) {
                            res.add(0, nums[i]);
                        }

                        result.addAll(subProblemResult);
                    }
                }
            }

            return result;
        }

        public List<List<Integer>> twoSumOnSorted(int[] nums, int i, int j, int targetSum) {
            List<List<Integer>> result = new ArrayList<List<Integer>>();

            while (i < j) {
                int sum = nums[i] + nums[j];
                if (sum == targetSum) {
                    List<Integer> res = new ArrayList<>(Arrays.asList(nums[i], nums[j]));
                    result.add(res);
                    i++;
                    j--;
                    // skip dupicates
                    while (i < j && nums[i] == nums[i - 1])
                        i++;
                    while (i < j && nums[j] == nums[j + 1])
                        j--;
                } else if (sum < targetSum) {
                    i++;
                } else {
                    j--;
                }
            }

            return result;
        }
    }

    class MatrixOps {

        public void setZeroes(int[][] matrix) {
            if (matrix.length == 0) {
                return;
            }
            int n = matrix.length;
            int m = matrix[0].length;
            boolean[] rows = new boolean[n];
            boolean[] cols = new boolean[m];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    if (matrix[i][j] == 0) {
                        rows[i] = true;
                        cols[j] = true;
                    }
                }
            }

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    if (rows[i] || cols[j]) {
                        matrix[i][j] = 0;
                    }
                }
            }
        }

        /*
         * You are given an n x n 2D matrix representing an image, rotate the image by
         * 90 degrees (clockwise).
         * 
         * Input: matrix = [[1,2,3],[4,5,6],[7,8,9]] Output: [[7,4,1],[8,5,2],[9,6,3]]
         * 
         * clockwise rotate first reverse up to down, then swap the symmetry 1 2 3 7 8 9
         * 7 4 1 4 5 6 => 4 5 6 => 8 5 2 7 8 9 1 2 3 9 6 3
         * 
         * 
         * anticlockwise rotate first reverse left to right, then swap the symmetry 1 2
         * 3 3 2 1 3 6 9 4 5 6 => 6 5 4 => 2 5 8 7 8 9 9 8 7 1 4 7
         */
        public void rotateImageClockWise(int[][] matrix) {
            int n = matrix.length;
            // reverse the order of rows top to bottom
            for (int row = 0; row < n / 2; row++) {
                for (int col = 0; col < n; col++) {
                    int temp = matrix[row][col];
                    matrix[row][col] = matrix[n - row - 1][col];
                    matrix[n - row - 1][col] = temp;
                }
            }

            // rerverse numbers in each diagonal top to bottom
            // this is same as swpping symmetric positions
            for (int row = 0; row < n; row++) {
                for (int col = row + 1; col < n; col++) {
                    int temp = matrix[row][col];
                    matrix[row][col] = matrix[col][row];
                    matrix[col][row] = temp;
                }
            }
        }

        public boolean checkDuplicateWithinK(int[] a, int k) {
            int n = a.length;
            k = Math.min(n, k);

            Set<Integer> slidingWindow = new HashSet<Integer>(k);

            // create initial wiindow of size k
            int i;
            for (i = 0; i < k; i++) {
                if (slidingWindow.contains(a[i])) {
                    return true;
                }

                slidingWindow.add(a[i]);
            }

            // now slide
            for (i = k; i < n; i++) {
                slidingWindow.remove(a[i - k]);
                if (slidingWindow.contains(a[i])) {
                    return true;
                }
                slidingWindow.add(a[i]);
            }

            return false;
        }

        public boolean checkDuplicateWithinK(int[][] mat, int k) {
            class Cell {
                int row;
                int col;

                public Cell(int r, int c) {
                    this.row = r;
                    this.col = c;
                }
            }

            int n = mat.length;
            int m = mat[0].length;
            k = Math.min(k, n * m);

            // map from distance to cell postions of the matrix
            Map<Integer, Set<Cell>> slidingWindow = new HashMap<Integer, Set<Cell>>();

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    if (slidingWindow.containsKey(mat[i][j])) {
                        for (Cell c : slidingWindow.get(mat[i][j])) {
                            int manhattanDist = Math.abs(i - c.row) + Math.abs(j - c.col);

                            if (manhattanDist <= k) {
                                return true;
                            }

                            if (i - c.row > k) {
                                slidingWindow.remove(c);
                            }
                        }

                        slidingWindow.get(mat[i][j]).add(new Cell(i, j));
                    } else {
                        slidingWindow.put(mat[i][j], new HashSet<Cell>());
                        slidingWindow.get(mat[i][j]).add(new Cell(i, j));
                    }
                }
            }

            return false;
        }

        /**
         * Write an efficient algorithm that searches for a value in an m x n matrix. 
         * This matrix has the following properties:
         * Integers in each row are sorted in ascending from left to right.
         * Integers in each column are sorted in ascending from top to bottom.
         * 
         * @param matrix
         * @param target
         * @return
         */
        public boolean searchMatrix2(int[][] matrix, int target) {
            if (matrix == null || matrix.length == 0) {
                return false;
            }
            int n = matrix.length;
            int m = matrix[0].length;
            int r = m - 1;
            int t = 0;

            while (t < n && r >= 0) {
                if (matrix[t][r] == target) {
                    return true;
                } else if (matrix[t][r] > target) {
                    r--;
                } else {
                    t++;
                }
            }

            return false;
        }

        public boolean searchMatrix1(int[][] matrix, int target) {
            if (matrix == null || matrix.length == 0) {
                return false;
            }

            // find the row
            int n = matrix.length;
            int m = matrix[0].length;

            int row = -1;
            int rl = 0;
            int rh = n - 1;
            while (rl <= rh) {
                int mid = (rl + rh) / 2;
                if (matrix[mid][0] == target || matrix[mid][m - 1] == target) {
                    return true;
                } else if (matrix[mid][0] > target) {
                    rh = mid - 1;
                } else {
                    if (matrix[mid][m - 1] > target) {
                        row = mid;
                        break;
                    } else {
                        rl = mid + 1;
                    }
                }
            }

            if (row == -1) {
                return false;
            }

            // search in the row
            int col = -1;
            int cl = 0;
            int ch = m - 1;
            while (cl <= ch) {
                int mid = (cl + ch) / 2;
                if (matrix[row][mid] == target) {
                    return true;
                } else if (matrix[row][mid] > target) {
                    ch = mid - 1;
                } else {
                    cl = mid + 1;
                }
            }

            return false;
        }
        
        /**
         * Given a binary matrix mat[n][n], find k such that all elements in k’th row are 0 and 
         * all elements in k’th column are 1. The value of mat[k][k] can be anything (either 0 or 1). 
         * If no such k exists, return -1.

            Examples:
            
            Input: mat[n][n] = {{0, 1, 1, 0, 1},
            {0, 0, 0, 0, 0},
            {1, 1, 1, 0, 0},
            {1, 1, 1, 1, 0},
            {1, 1, 1, 1, 1}};
            Output: 1
            All elements in 1’st row are 0 and all elements in
            1’st column are 1. mat[1][1] is 0 (can be any value)
            
            Input: mat[n][n] = {{0, 1, 1, 0, 1},
            {0, 0, 0, 0, 0},
            {1, 1, 1, 0, 0},
            {1, 0, 1, 1, 0},
            {1, 1, 1, 1, 1}};
            Output: -1
            There is no k such that k’th row elements are 0 and
            k’th column elements are 1.            
         * @param mat
         * @return
         */
        public int findKthRowCol(int mat[][]){
            int n = mat.length;
            int m = mat[0].length;
            int i = 0;
            int j = 0;
            
            int candidate = -1;
            while(i < n && j < m){
                //check the row for all zero
                if(mat[i][j] == 0){
                    int k = j+1;
                    while(k < m && mat[i][k] == 0){
                        k++;
                    }
                    
                    if(k == m){
                        candidate = i;
                        break;
                    }
                    //if not all zero in this row, then this row can't be the candidate
                    else{
                        i++;
                    }
                }
                //check the column for all ones
                else{
                    int k = i+1;
                    while(k < n && mat[k][j] == 0){
                        k++;
                    }
                    
                    if(k == n){
                        candidate = i;
                        break;
                    }
                    //if not all are 1 then this col can't be the candidate
                    else{
                        j++;
                    }
                }
            }
            
            //we found a row/cold candidate, validate the rowand columnd
            if(candidate != -1){
                for(j = 0; j<n; j++){
                    if(j != candidate && mat[candidate][j] != 0){
                        return -1;
                    }
                }
                
                for(i = 0; i<n; i++){
                    if(i != candidate && mat[i][candidate] != 1){
                        return -1;
                    }
                }
                
                return candidate;
            }
            
            return candidate;
        }

        public int largestPlusInMatrix(int M[][]) {
            int n = M.length;
            int m = M[0].length;
            int left[][] = new int[n + 2][m + 2];
            int right[][] = new int[n + 2][m + 2];
            int top[][] = new int[n + 2][m + 2];
            int bottom[][] = new int[n + 2][m + 2];

            // topdown
            for (int i = 1; i <= n; i++) {
                for (int j = 1; j <= m; j++) {
                    left[i][j] = (M[i - 1][j - 1] == 0) ? 0 : left[i][j - 1] + 1;
                    top[i][j] = (M[i - 1][j - 1] == 0) ? 0 : top[i - 1][j] + 1;
                }
            }

            // bottomup
            for (int i = n; i >= 1; i--) {
                for (int j = m; j >= 1; j--) {
                    right[i][j] = (M[i - 1][j - 1] == 0) ? 0 : right[i][j + 1] + 1;
                    bottom[i][j] = (M[i - 1][j - 1] == 0) ? 0 : bottom[i + 1][j] + 1;
                }
            }

            int minPlus[][] = new int[n][m];
            int maxPlusLen = -1;
            int maxPlusRow = -1;
            int maxPlusCol = -1;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    minPlus[i][j] = Math.min(Math.min(left[i + 1][j + 1], right[i + 1][j + 1]),
                            Math.min(top[i + 1][j + 1], bottom[i + 1][j + 1]));

                    if (minPlus[i][j] > maxPlusLen) {
                        maxPlusLen = minPlus[i][j];
                        maxPlusRow = i;
                        maxPlusCol = j;
                    }
                }
            }

            System.out.println("[row,col]=[" + maxPlusRow + "," + maxPlusCol + "]");
            return (maxPlusLen - 1) * 4 + 1;
        }
    }

    class Anagrams {

        int[] primes = new int[] { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
                83, 89, 97, 101 };

        private boolean equalHistogram(int[] hist1, int[] hist2) {
            for (int i = 0; i < hist1.length; i++) {
                if (hist1[i] != hist2[i]) {
                    return false;
                }
            }

            return true;
        }

        public int hash(String s) {
            int hash = 1;
            for (int i = 0; i < s.length(); i++) {
                hash *= primes[s.charAt(i) - 'a'];
            }

            return hash;
        }

        // Input: strs = ["eat","tea","tan","ate","nat","bat"]
        // Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
        public List<List<String>> groupAnagrams(String[] strs) {
            List<List<String>> result = new LinkedList<List<String>>();

            Map<Integer, List<String>> histMap = new HashMap<>();

            for (String s : strs) {
                int hist = hash(s);

                List<String> res = histMap.get(hist);
                if (res == null) {
                    res = new LinkedList<String>();
                    histMap.put(hist, res);
                    result.add(res);
                }

                res.add(s);
            }

            return result;
        }

        public int searchAnagramSubstring(String text, String pattern) {
            int count = 0;
            int n = text.length();
            int m = pattern.length();

            if (n < m | m == 0 | m == 0) {
                return 0;
            }

            int textHist[] = new int[256];
            int patHist[] = new int[256];

            // initial histogram window of size m
            int i = 0;
            for (i = 0; i < m; i++) {
                patHist[pattern.charAt(i)]++;
                textHist[text.charAt(i)]++;
            }

            // search the pattern histogram in a sliding window of text histogram
            do {
                // O(1) time check as array size is constant
                if (equalHistogram(textHist, patHist)) {
                    System.out.println("anagram found : " + text.substring(i - m, i));
                    count++;
                }

                // slide the text histogram window by 1 position to the right and check for
                // anagram
                textHist[text.charAt(i)]++;
                textHist[text.charAt(i - m)]--;
            } while (++i < n);

            return count;
        }

        public class SearchAnagramInDictionary {

            private Set<String> dictionary = new HashSet<String>();

            public void add(String word) {
                dictionary.add(word);
            }

            public void addAll(List<String> words) {
                dictionary.addAll(words);
            }

            public boolean remove(String word) {
                return dictionary.remove(word);
            }

            private String getKey(String str) {
                str = str.toLowerCase().trim();
                int[] hist = new int[256];
                for (int i = 0; i < str.length(); i++) {
                    hist[str.charAt(i)]++;
                }

                StringBuilder sb = new StringBuilder();

                for (int val : hist) {
                    sb.append(val);
                }

                return sb.toString();
            }

            public int searchAnagram(String pattern) {
                int count = 0;
                HashMap<String, List<String>> histogramMap = new HashMap<String, List<String>>();

                for (String word : dictionary) {
                    String key = getKey(word);

                    if (!histogramMap.containsKey(key)) {
                        histogramMap.put(key, new ArrayList<String>());
                    }

                    histogramMap.get(key).add(word);
                }

                String searchKey = getKey(pattern);
                List<String> res = histogramMap.get(searchKey);

                if (res != null) {
                    count = res.size();

                    System.out.print("anagrams in dict: ");
                    for (String s : res) {
                        System.out.print(s + " ");
                    }
                    System.out.println();
                }

                return count;
            }
        }
    }

    class LongestPalindrome {

        int startIndex = 0, max = 0;

        // faster but not intuitive
        public String longestPalindrome(String s) {
            if (s.length() < 2)
                return s;
            char[] c = s.toCharArray();
            int i = 0;
            while (i < c.length) {
                // System.out.println(i);
                i = checkPalin(c, i);
            }

            return s.substring(startIndex, startIndex + max);
        }

        public int checkPalin(char[] c, int i) {
            int end = i + 1;

            // skip all the duplicate charas on right as duplicates character substrings are
            // by default palindrom
            while (end < c.length && c[i] == c[end])
                end++;
            // suggesting caller where to start in next call
            int next = end;

            int start = i - 1;
            // now extend to left of i such that the right of end (next) is extend
            // any mismath should stop extending
            while (start > -1 && end < c.length) {
                if (c[start] == c[end]) {
                    start--;
                    end++;
                } else
                    break;
            }

            // now we have a localMaxima palindrom of length end - start + 1
            // update global maxima
            if (end - start - 1 > max) {
                max = end - start - 1;
                startIndex = ++start;
            }

            return next;

        }
        
        public boolean isPalindrome(String s) {
            if(s == null){
                return false;
            }
            int i = 0, j = s.length() - 1;
            while(i < j){
                while(i < s.length() && !Character.isLetterOrDigit(s.charAt(i))) i++;
                while(j >= 0 && !Character.isLetterOrDigit(s.charAt(j))) j--;
                
                if(i < s.length() && j >= 0 && Character.toLowerCase(s.charAt(i)) != Character.toLowerCase(s.charAt(j)))
                    return false;
                i++;
                j--;
            }
            return true;
        }

        // intuitive
        // for each index extend to both side to either get a even length palindrom with
        // center at space between i-1 and i
        // or extend to both side to get a even length palindrom with center at i
        public String longestPalindromN2(String str) {
            int n = str.length();
            if (n <= 1) {
                return str;
            }

            int l = 0;
            int h = 0;
            int start = 0;
            int maxlen = 1;

            for (int i = 1; i < n; i++) {
                // palindrom of even length with center in space between i-1 and i
                l = i - 1;
                h = i;
                int len = 0;
                while (l >= 0 && h <= n - 1 && (str.charAt(l) == str.charAt(h))) {
                    len = h - l + 1;
                    if (len > maxlen) {
                        start = l;
                        maxlen = len;
                    }
                    l--;
                    h++;
                }

                // palindrom of odd length with center at i
                l = i;
                h = i;
                while (l >= 0 && h <= n - 1 && (str.charAt(l) == str.charAt(h))) {
                    len = h - l + 1;
                    if (len > maxlen) {
                        start = l;
                        maxlen = len;
                    }
                    l--;
                    h++;
                }
            }

            return str.substring(start, start + maxlen);
        }

        public int longestPalindromDP(String str) {
            int n = str.length();
            int dp[][] = new int[n + 1][n + 1];
            for (int i = 1; i < n; i++) {
                dp[i][i] = 1;
            }

            // find palindrom of each possible length
            for (int len = 2; len <= n; len++) {
                // try to get a palindrom of length len starting at each position i
                for (int i = 1; i <= n - len + 1; i++) {
                    // right end position of current palindrom
                    int j = i + len - 1;

                    if (str.charAt(i - 1) == str.charAt(j - 1)) {
                        dp[i][j] = 2 + dp[i + 1][j - 1];
                    } else {
                        dp[i][j] = Math.max(dp[i][j - 1], dp[i + 1][j]);
                    }
                }
            }

            return dp[1][n];
        }

        public int minInsertionsForLongestPalindrom(final String str) {
            final int n = str.length();
            // L[i][j] contains minimum number of deletes required to make string(i..j) a
            // palindrome
            final int[][] L = new int[n][n];

            // find L for each pair of increasing range (i,j) where i<=j. That is we are
            // only populating upperhalf of the
            // table
            for (int i = 1; i < n; i++) {
                for (int j = i, k = 0; j < n; j++, k++) {
                    // if characters are same at the two boundary then no deletions required in
                    // these positions.
                    // if characters are not same the we can insert either string[i] at the gap
                    // between j-1 and j or we
                    // can insert string[j] at the gap between i and i+1. We take the min of these
                    L[k][j] = str.charAt(k) == str.charAt(j) ? L[k + 1][j - 1] : Math.min(L[k][j - 1], L[k + 1][j]) + 1;
                }
            }

            return L[0][n - 1];
        }
    }

    
    class Subsequences {
        /**
         * Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.
         * Input: nums = [100,4,200,1,3,2]
         * Output: 4
         * Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
         * @param nums
         * @return
         */
        public int longestConsecutiveSequence(int[] nums) {
            if(nums.length == 0){
                return 0;
            }
            Arrays.sort(nums);
            
            int maxLen = 1, len = 1;
            for(int i = 1; i < nums.length; i++){
                // if same number is repeating we need only one to count
                if(nums[i] == nums[i-1]){
                    continue;
                }
                // increanse the subseq only if consecutive
                if(nums[i] == nums[i-1]+1){
                    maxLen = Math.max(maxLen, ++len);
                }
                // otherwise start a new sequence
                else{
                    len = 1;
                }
            }
            
            return maxLen;
        }
        
        public int lengthOfLIS(int[] nums) {
            if (nums == null || nums.length == 0) {
                return 0;
            }
            int[] lis = new int[nums.length];
            lis[0] = 1;
            int max = 1;

            for (int i = 1; i < nums.length; i++) {
                lis[i] = 1;
                for (int j = 0; j < i; j++) {
                    if (nums[i] > nums[j]) {
                        lis[i] = Math.max(lis[i], 1 + lis[j]);
                        max = Math.max(max, lis[i]);
                    }
                }
            }

            return max;
        }

        public int findNumberOfLIS(int[] nums) {
            if (nums == null || nums.length == 0) {
                return 0;
            }
            int[] lis = new int[nums.length];
            int max = 0;
            int res = 0;
            int counts[] = new int[nums.length];

            for (int i = 0; i < nums.length; i++) {
                lis[i] = 1;
                counts[i] = 1;
                for (int j = 0; j < i; j++) {
                    if (nums[i] > nums[j]) {
                        // if another prefix exists that makes same length subseq
                        // including current number then add the counts
                        if (lis[i] == lis[j] + 1) {
                            counts[i] += counts[j];
                        }
                        // if a longer prefix exists that makes more length subseq
                        // including current number them reset the count
                        else if (lis[i] < lis[j] + 1) {
                            lis[i] = lis[j] + 1;
                            counts[i] = counts[j];
                        }
                    }
                }

                // set the longest subseq and update the result accordingly
                if (lis[i] > max) {
                    max = lis[i];
                    res = counts[i];
                } else if (lis[i] == max) {
                    res += counts[i];
                }
            }

            return res;
        }
        
        /**
         * Given a sequence as as array of positive integers. Find the length of longest bitonic subsequence. 
         * A bitonic subsequence is a subsequence that is first increasing up to a peak value and then 
         * decreasing from the peak value. For example, A=[1, 11, 2, 10, 4, 5, 2, 1] the longest bitonic 
         * sequence is 1, 2, 4, 5, 2, 1 of length 6. For A=[0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15] 
         * longest bitonic sequence is 0, 8, 12, 14, 13, 11, 7 of length 7.
         * 
         * 
                          LIS     peak    LDS
                           |       |       |
                           |       v       |
                           |      14       |
                           v   12    13    v
                             8         11
                           0             7
         * 
         * LIS[i] : length of the Longest Increasing subsequence ending at arr[i]. 
         * LDS[i]:  length of the longest Decreasing subsequence starting from arr[i].         
         * LIS[i]+LDS[i]-1 : the length Longest Bitonic Subsequence with peak at i.
         * 
         * LIS(i) = max{1+LIS(j)} for all j < i and A[j] < A[i] 
         * 
         *     
                                     lis(4)           
                                 /       |      \
                         lis(3)      lis(2)    lis(1)  
                        /     \        /         
                  lis(2)  lis(1)   lis(1) 
                  /    
                lis(1) 

               So, the LIS problem has an optimal substructure LIS(i) = max{1+LIS(j)} for all j < i 
               and the subproblems are repeating.

         * 
         * @param a
         * @return
         */
        public int longestBiotonicSequence(int[] a){
            int[] lis = new int[a.length];
            int[] lds = new int[a.length];
            //base cases - single number is a lis and lds
            Arrays.fill(lis, 1);
            Arrays.fill(lds, 1);
            int maxBiotonicLen = Integer.MIN_VALUE;
            
            //longest increasing subsequence
            //lis(i) = max{1+lis(j)}, for all j < i and a[j] < a[i]
            for(int i = 1; i < a.length; i++){
                for(int j = 0; j < i; j++){
                    if(a[i] > a[j] && lis[j] + 1 > lis[i]){
                        lis[i] = lis[j]+1;
                    }
                }
            }
            
            //longest decreasing subsequence
            //lds(i) = max{1+lds(j)}, for all j < i and a[j] > a[i]
            //longest biotonic seq lbs(i) = lis(i)+lds(i)-1
            maxBiotonicLen = lis[0]+lds[0]-1;
            for(int i = 1; i < a.length; i++){
                for(int j = 0; j < i; j++){
                    if(a[i] < a[j] && lds[j] + 1 > lds[i]){
                        lds[i] = lds[j]+1;
                    }
                }
                maxBiotonicLen = Math.max(maxBiotonicLen, lis[i]+lds[i]-1);
            }
            
            return maxBiotonicLen;
        }

        public boolean increasingTripletSubseq(int[] a) {
            int[] lis = new int[a.length];
            // base cases - single number is a lis and lds
            Arrays.fill(lis, 1);
            // longest increasing subsequence
            // lis(i) = max{1+lis(j)}, for all j < i and a[j] < a[i]
            for (int i = 1; i < a.length; i++) {
                for (int j = 0; j < i; j++) {
                    if (a[i] > a[j] && lis[j] + 1 > lis[i]) {
                        lis[i] = lis[j] + 1;
                        if (lis[i] >= 3) {
                            return true;
                        }
                    }
                }
            }

            return false;
        }
        
        /**
         *     Give a list of unsorted number, find the min window or min sublist or min subarray 
         *     of the input, such as if sublist is sorted, the whole list is sorted too.
         *     
         *     For example, given array a={1,2,3,5,4,4,3,3,7,8,9} then min subarray to sort the complete 
         *     array sorted is {5,4,3,3}. 
         *     More example : 
         *     for a={1,2,3,5,6,4,2,3,3,7,8,9} then min subarray is {2,3,5,6,4,2,3,3}, 
         *     for a={1,2,3,5,6,4,3,3,7,8,9,2} then min subarray is {2,3,5,6,4,2,3,3,7,8,9,2} etc.
         * @param nums
         * @return
         */
        //O(n) time algorithm
        public List<Integer> minListToBeSorted(int[] nums){
            //find the first index from left to right where the sorted order disrupted
            //i.e. first index where next element is smaller 
            int minIndex = -1;
            for(int i = 1; i< nums.length; i++){
                if(nums[i] < nums[i-1]){
                    minIndex = i;
                    break;
                }
            }
            
            //So, we got a potential mid element of the unsorted list
            //the minimum list must have a minimum element which is smaller or equal to this element
            for(int i = minIndex; i<nums.length; i++){
                if(nums[i] < nums[minIndex]){
                    minIndex = i;
                }
            }
            
            //we can use the min element to identify the left boundary of the list because the left boundary
            //is the first element to the left greater than equal to this smallest element.
            //at the same time we can compute the maximum element in the left of this min element
            int l = minIndex;
            int r = minIndex;
            int maxLeft = nums[l];
            while(l >= 0 && nums[l] >= nums[minIndex]){
                maxLeft = Math.max(maxLeft, nums[l]);
                l--;
            }
            
            //we can use the max element to find the right most boundary of the min unsorted list by finding
            //first node in right of the smallest element that is greater than the max element
            while(r < nums.length && nums[r] <= maxLeft){
                r++;
            }
            
            //all elments between 
            List<Integer> res = new  ArrayList<Integer>();
            for(int i = l+1; l>=0 && r<=nums.length && i<r; i++){
                res.add(nums[i]);
            }
            return res;
        }
    }

    class IntervalOps {

        public int findLongestChain(int[][] pairs) {
            Arrays.sort(pairs, (a, b) -> (a[1] - b[1])); // Assume that Pair class implements comparable with the
                                                         // compareTo() method such that (a, b) < (c,d) iff b<c
            int chainLength = 0;

            // select the first pair of the sorted pairs array
            chainLength++;
            int prev = 0;

            for (int i = 1; i < pairs.length; i++) {
                if (pairs[i][0] > pairs[prev][1]) {
                    chainLength++;
                    prev = i;
                }
            }
            return chainLength;
        }

        public int maxOverlapIntervalCount(int[] start, int[] end) {
            int maxOverlap = 0;
            int currentOverlap = 0;

            Arrays.sort(start);
            Arrays.sort(end);

            int i = 0;
            int j = 0;
            int m = start.length, n = end.length;
            while (i < m && j < n) {
                if (start[i] < end[j]) {
                    currentOverlap++;
                    maxOverlap = Math.max(maxOverlap, currentOverlap);
                    i++;
                } else {
                    currentOverlap--;
                    j++;
                }
            }

            return maxOverlap;
        }

        public int[][] mergeOverlappedIntervals(int[][] intervals) {
            if (intervals.length == 0) {
                return new int[0][0];
            }

            // O(nlgn) sort based on start
            Arrays.sort(intervals, (a, b) -> a[0] - b[0]);

            LinkedHashSet<int[]> res = new LinkedHashSet<>();
            int[] prev = intervals[0];

            // O(n) for overlaps
            for (int i = 1; i < intervals.length; i++) {
                // overlaps
                if (intervals[i][0] <= prev[1]) {
                    // merge intervals
                    prev[1] = Math.max(prev[1], intervals[i][1]);
                }
                // current interval ends
                else {
                    res.add(prev);
                    prev = intervals[i];
                }
            }

            if (!res.contains(prev)) {
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
            for (int i = 1; i < intervals.length; i++) {
                // overlaps
                if (intervals[i].start < prev.end) {
                    // merge intervals
                    prev.end = Math.max(prev.end, intervals[i].end);
                }
                // current interval ends
                else {
                    res.add(prev);
                    prev = intervals[i];
                }
            }

            if (!res.contains(prev)) {
                res.add(prev);
            }

            Interval[] res1 = new Interval[res.size()];

            return res.toArray(res1);
        }

        public int[][] insertInterval(int[][] intervals, int[] newInterval) {
            LinkedList<int[]> res = new LinkedList<>();

            for (int[] curInterval : intervals) {
                // if new interval completely left to current then take the in the result
                // make the current as new interval as it's position has been taken
                if (curInterval[0] > newInterval[1]) {
                    res.add(newInterval);
                    newInterval = curInterval;
                }
                // if new interval is completely right ogf the current then take the current in
                // result
                // new interval remain same with the hope the next interval in input may merge
                // it
                else if (curInterval[1] < newInterval[0]) {
                    res.add(curInterval);
                }
                // otherewise merge
                else {
                    newInterval = new int[] { Math.min(curInterval[0], newInterval[0]),
                            Math.max(curInterval[1], newInterval[1]) };
                }
            }

            res.add(newInterval);

            return res.toArray(new int[res.size()][]);
        }

        public Interval[] insertInterval(Interval[] intervals, Interval newInterval) {
            Set<Interval> res = new LinkedHashSet<>();

            for (Interval curInterval : intervals) {
                // if new interval completely left to current then take the in the result
                // make the current as new interval as it's position has been taken
                if (curInterval.start > newInterval.end) {
                    res.add(newInterval);
                    newInterval = curInterval;
                }
                // if new interval is completely right of the current then take the current in
                // result
                // new interval remain same with the hope the next interval in input may merge
                // it
                else if (curInterval.end < newInterval.start) {
                    res.add(curInterval);
                }
                // otherewise merge
                else {
                    newInterval = new Interval(Math.min(curInterval.start, newInterval.start),
                            Math.max(curInterval.end, newInterval.end));
                }
            }

            res.add(newInterval);

            Interval[] ret = new Interval[res.size()];
            return res.toArray(ret);
        }

        public int[][] intervalIntersection(int[][] A, int[][] B) {
            if (A == null || B == null || A.length == 0 || B.length == 0) {
                return new int[][] {};
            }

            int i = 0, j = 0;
            int m = A.length;
            int n = B.length;
            List<int[]> res = new ArrayList<>();

            while (i < m && j < n) {
                // intersection possible if one's start within another range
                if ((A[i][0] <= B[j][1]) && (B[j][0] <= A[i][1])) {
                    // intersection is the max start and minium end between A[i] and B[j]
                    res.add(new int[] { Math.max(A[i][0], B[j][0]), Math.min(A[i][1], B[j][1]) });
                }

                // A[i] is on left or equal of B[j] - move i
                if (A[i][1] <= B[j][1]) {
                    i++;
                }
                // B[j] is on left of A[i] - move j
                else {
                    j++;
                }
            }

            int[][] ret = new int[res.size()][2];
            return res.toArray(ret);
        }

        /**
         * A string S of lowercase English letters is given. We want to partition this string 
         * into as many parts as possible so that each letter appears in at most one part, 
         * and return a list of integers representing the size of these parts.

            Example 1:
            
            Input: S = "ababcbacadefegdehijhklij"
            Output: [9,7,8]
            Explanation:
            The partition is "ababcbaca", "defegde", "hijhklij".
            This is a partition so that each letter appears in at most one part.
            A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits S into less parts.
            
         * @author rahzahid
         *
         */
        public class PartitionLabels {
            public class Interval {
                public int start;
                public int end;

                public Interval(int start, int end) {
                    this.start = start;
                    this.end = end;
                }
            }

            public List<Integer> partitionLabels(String S) {
                if (S == null || S.length() == 0) {
                    return null;
                }
                // one interval per character
                // insert order is based on first appearance of the character
                LinkedHashMap<Integer, Interval> map = new LinkedHashMap<Integer, Interval>(26);
                // build the intervalMap
                for (int i = 0; i < S.length(); i++) {
                    int c = S.charAt(i) - 'a';
                    if (!map.containsKey(c)) {
                        map.put(c, new Interval(i, i));
                    } else {
                        map.get(c).end = i;
                    }
                }

                LinkedHashSet<Interval> res = new LinkedHashSet<>();
                List<Integer> result = new ArrayList<>();
                Iterator it = map.entrySet().iterator();
                Map.Entry<Integer, Interval> entry = (Map.Entry<Integer, Interval>) it.next();
                Interval prev = entry.getValue();

                // now start merging intervals
                while (it.hasNext()) {
                    entry = (Map.Entry<Integer, Interval>) it.next();
                    Interval cur = entry.getValue();

                    if (cur.start < prev.end) {
                        prev = new Interval(prev.start, Math.max(prev.end, cur.end));
                    } else {
                        res.add(prev);
                        result.add(prev.end - prev.start + 1);
                        prev = cur;
                    }
                }

                if (!res.contains(prev)) {
                    res.add(prev);
                    result.add(prev.end - prev.start + 1);
                }

                return result;
            }
        }

        class ActivitySelection {

            public int activitySelecyion(int[][] pairs) {
                Arrays.sort(pairs); // Assume that Pair class implements comparable with the compareTo() method such
                                    // that (a, b) < (c,d) iff b<c
                int chainLength = 0;

                // select the first pair of the sorted pairs array
                chainLength++;
                int prev = 0;

                for (int i = 1; i < pairs.length; i++) {
                    if (pairs[i][0] >= pairs[prev][1]) {
                        chainLength++;
                        prev = i;
                    }
                }
                return chainLength;
            }

            public int weightedActivitySelection(Job[] jobs) {
                int n = jobs.length;
                int profit[] = new int[n + 1];
                int q[] = new int[n];

                // sort according to finish time
                Arrays.sort(jobs);

                // find q's - O(nlgn)
                for (int i = 0; i < n; i++) {
                    q[i] = binarySearchLatestCompatibleJob(jobs, 0, n - 1, jobs[i].start);
                }

                // compute optimal profits - O(n)
                profit[0] = 0;
                for (int j = 1; j <= n; j++) {
                    int profitExcluding = profit[j - 1];
                    int profitIncluding = jobs[j - 1].weight;
                    if (q[j - 1] != -1) {
                        profitIncluding += profit[q[j - 1] + 1];
                    }
                    profit[j] = Math.max(profitIncluding, profitExcluding);
                }
                return profit[n];
            }

            public int binarySearchLatestCompatibleJob(Job[] A, int l, int h, int key) {
                int mid = (l + h) / 2;

                if (A[h].finish <= key) {
                    return h;
                }
                if (A[l].finish > key) {
                    return -1;
                }

                if (A[mid].finish == key) {
                    return mid;
                }
                // mid is greater than key, so floor is either mid-1 or it exists in A[l..mid-1]
                else if (A[mid].finish > key) {
                    if (mid - 1 >= l && A[mid - 1].finish <= key) {
                        return mid - 1;
                    } else {
                        return binarySearchLatestCompatibleJob(A, l, mid - 1, key);
                    }
                }
                // mid is less than key, so floor is either mid or it exists in A[mid+1....h]
                else {
                    if (mid + 1 <= h && A[mid + 1].finish > key) {
                        return mid;
                    } else {
                        return binarySearchLatestCompatibleJob(A, mid + 1, h, key);
                    }
                }
            }
        }

        public int maxLoad(Job[] jobs) {
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
                    return Integer.compare(o1.finish, o2.finish);
                }
            });

            int i = 0, j = 0;
            while (i < start.length && j < end.length) {
                if (start[i].start <= end[j].finish) {
                    curLoad += start[i].weight;
                    maxLoad = Math.max(maxLoad, curLoad);
                    i++;
                } else {
                    curLoad -= end[j].weight;
                    j++;
                }
            }

            return maxLoad;
        }

        public ArrayList<Strip> skyLine(Building[] buildings) {
            int n = buildings.length;
            Building[] start = Arrays.copyOf(buildings, n);
            Building[] end = Arrays.copyOf(buildings, n);
            PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(n, Collections.reverseOrder());
            ArrayList<Strip> strips = new ArrayList<Strip>();

            // sort based on left coordinate of a building i.e. start of a range
            Arrays.sort(start, new Comparator<Building>() {

                @Override
                public int compare(Building o1, Building o2) {
                    int c = Integer.compare(o1.l, o2.l);
                    if (c == 0) {
                        c = Integer.compare(o2.h, o1.h);
                    }
                    return c;
                }
            });

            // sort based on right coordinate of a building i.e. end of a range
            Arrays.sort(end, new Comparator<Building>() {

                @Override
                public int compare(Building o1, Building o2) {
                    return Integer.compare(o1.r, o2.r);
                }
            });

            int i = 0, j = 0;
            while (i < n || j < n) {
                // a new overlapping range i.e. a building
                if (i < n && start[i].l <= end[j].r) {
                    // update max height seen so far in current overlap
                    maxHeap.add(start[i].h);
                    // max height in current overlap including the current building
                    int maxHeightIncldingMe = maxHeap.isEmpty() ? 0 : maxHeap.peek();
                    // add th current strip with the left of building and max height seen so far in
                    // currne overlap
                    strips.add(new Strip(start[i].l, maxHeightIncldingMe));
                    // try next building
                    i++;
                } else {
                    // it's an end of a range of current overlap. So, we need to remove the height
                    // of this range i.e. building from the max heap
                    maxHeap.remove(end[j].h);
                    // max height of remaining buildings in current overlap
                    int maxHeightExcldingMe = maxHeap.isEmpty() ? 0 : maxHeap.peek();
                    // add the current strip with the right of building and max height of remaining
                    // buildings
                    strips.add(new Strip(end[j].r, maxHeightExcldingMe));
                    // update end index
                    j++;
                }
            }

            // merge strips to remove successive strips with same height
            ArrayList<Strip> mergedStrips = new ArrayList<Strip>();
            int prevHeight = 0;
            for (Strip st : strips) {
                if (st.l == end[n - 1].r && st.h != 0) {
                    continue;
                }
                if (prevHeight == 0) {
                    prevHeight = st.h;
                    mergedStrips.add(st);
                } else if (prevHeight != st.h) {
                    prevHeight = st.h;
                    mergedStrips.add(st);
                }
            }

            return mergedStrips;
        }
    }

    public static class BlockingQueue implements Serializable {

        private static final long serialVersionUID = 1L;

        // thread safe instance - threads always read the latest updates - read "happens
        // after" write
        private static volatile BlockingQueue blockingQueueInstance = null;

        private Queue<Integer> queue;
        private ReentrantLock lock = new ReentrantLock();
        private Condition notEmptyCondition = lock.newCondition();
        private Condition notFullCondition = lock.newCondition();
        private int capacity;

        // making it singleton
        private BlockingQueue(int capacity) {
            if (blockingQueueInstance != null) {
                throw new RuntimeException("SingleTone viloation: use getInstance method to get the instance");
            }

            queue = new LinkedList<>();
            this.capacity = capacity;
        }

        // create single instance or return existing
        public static BlockingQueue getInstance(int capacity) {
            // check ti make sure instance not created
            if (blockingQueueInstance == null) {
                // another thread may also try to instantiate at same time
                // so put under a monitor
                synchronized (BlockingQueue.class) {
                    // double check to make sure
                    if (blockingQueueInstance == null) {
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
                if (queue.size() == capacity) {
                    notFullCondition.await();
                }

                queue.add(data);
                // as we added some we can notify that queue is not empty anymore
                notEmptyCondition.notifyAll();
            } finally {
                lock.unlock();
            }
        }

        public boolean offer(int data, long timeout, TimeUnit unit) throws Exception {
            // take the reentrant lock
            lock.lock();
            lock.lockInterruptibly();

            try {
                while (queue.size() == capacity) {
                    if (timeout <= 0)
                        return false;
                    notFullCondition.await(timeout, unit);
                }

                queue.add(data);
                // as we added some we can notify that queue is not empty anymore
                notEmptyCondition.notifyAll();
            } finally {
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
                if (queue.isEmpty()) {
                    notEmptyCondition.await();
                }

                res = queue.remove();
                // as we removed some we can notify that queue is not full anymore
                notFullCondition.notifyAll();
            } finally {
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
                while (queue.isEmpty()) {
                    if (timeout <= 0)
                        return null;
                    notEmptyCondition.await(timeout, unit);
                }

                res = queue.remove();
                // as we removed some we can notify that queue is not full anymore
                notFullCondition.notifyAll();
            } finally {
                lock.unlock();
            }

            return res;
        }
    }

    class maxSumPath {

        int maxSum = Integer.MIN_VALUE;

        public int maxSumPath(TreeNode root) {
            maxSumPathDown(root);

            return maxSum;
        }

        public int maxSumPathDown(TreeNode root) {
            if (root == null) {
                return 0;
            }

            // max sum path down (not through root) can be either on left subtree or right
            // subtree
            int leftMaxSumPath =  Math.max(0, maxSumPathDown(root.left));
            int rightMaxSumPath =  Math.max(0, maxSumPathDown(root.right));

            // on the way compute the sum on the path through root and update the global max
            // this is because sums from either subtree can be negative
            // so going through root is not always the max sum path
            maxSum = Math.max(maxSum, leftMaxSumPath + rightMaxSumPath + root.val);

            // return the max one ending at root
            return Math.max(leftMaxSumPath, rightMaxSumPath) + root.val;
        }

        private int minLenSumPathBST(final TreeNode root, final int sum, final int len) {
            if (root == null) {
                return Integer.MAX_VALUE;
            }

            // find the remaining sum as we are including current node in the current path
            final int remainingSum = sum - root.val;
            // If remaining sum is zero and it is A leaf node then we found A complete path
            // from root to A leaf.
            if (remainingSum == 0 && root.left == null && root.right == null) {
                return len + 1;
            }
            // If remaining sum is less than current node value then we search remaining in
            // the left subtree.
            else if (remainingSum <= root.val) {
                int l = minLenSumPathBST(root.left, remainingSum, len + 1);
                // if search in left subtree fails to find such path only then we search in the
                // right subtree
                if (l == Integer.MAX_VALUE) {
                    l = minLenSumPathBST(root.right, remainingSum, len + 1);
                }

                return l;

            }
            // If remaining sum is greater than current node value then we search remaining
            // in the right subtree.
            else {
                int l = minLenSumPathBST(root.right, remainingSum, len + 1);
                // if search in right subtree fails to find such path only then we search in the
                // left subtree
                if (l == Integer.MAX_VALUE) {
                    l = minLenSumPathBST(root.left, remainingSum, len + 1);
                }

                return l;
            }
        }

        public int maxSumSubSeqNonContagious(int a[]) {
            int max_include[] = new int[a.length];
            int max_exclude[] = new int[a.length];
            max_include[0] = a[0];
            max_exclude[0] = Integer.MIN_VALUE;
            int max = a[0];

            for (int i = 1; i < a.length; i++) {
                max_include[i] = Math.max(max_exclude[i - 1] + a[i], a[i]);
                max_exclude[i] = Math.max(max_include[i - 1], max_exclude[i - 1]);
                max = Math.max(max_include[i], max_exclude[i]);
            }

            return max;
        }
    }

    class KthPermutation {

        public int permRank(final int[] X) {
            MergeSortApps m = new MergeSortApps();
            final int[] smaller_count = m.countSmallerOnRightWithMerge(X);

            final int[] factorial = new int[X.length];
            factorial[0] = 1;
            factorial[1] = 1;

            for (int i = 2; i < X.length; i++) {
                factorial[i] = i * factorial[i - 1];
            }

            int rank = 1;
            for (int i = 0; i < X.length; i++) {
                rank += smaller_count[i] * factorial[X.length - i - 1];
            }

            return rank;
        }

        /**
         * 1. Find the largest index k such that nums[k] < nums[k + 1]. 2. If no such
         * index exists, the permutation is sorted in descending order, just reverse it
         * to ascending order and we are done. For example, the next permutation of [3,
         * 2, 1] is [1, 2, 3]. 3. Find the largest index l greater than k such that
         * nums[k] < nums[l]. 4 Swap the value of nums[k] with that of nums[l]. 5.
         * Reverse the sequence from nums[k + 1] up to and including the fi
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
                reverse(nums, 0, nums.length - 1);
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
            reverse(nums, k + 1, nums.length - 1);
        }

        public String getPermutation(int n, int k) {
            int[] kthperm = kthPermutation(n, k);
            StringBuilder sb = new StringBuilder(kthperm.length);
            Arrays.stream(kthperm).forEach(i -> {
                sb.append(i);
            });

            return sb.toString();
        }

        public int[] kthPermutation(int n, int k) {
            final int[] nums = new int[n];
            final int[] factorial = new int[n + 1];

            factorial[0] = 1;
            factorial[1] = 1;
            nums[0] = 1;

            for (int i = 2; i <= n; i++) {
                nums[i - 1] = i;
                factorial[i] = i * factorial[i - 1];
            }

            if (k <= 1) {
                return nums;
            }
            if (k >= factorial[n]) {
                reverse(nums, 0, n - 1);
                return nums;
            }

            k -= 1;// 0-based
            for (int i = 0; i < n - 1; i++) {
                int fact = factorial[n - i - 1];
                // index of the element in the rest of the input set
                // to put at i position (note, index is offset by i)
                int index = (k / fact);
                // put the element at index (offset by i) element at position i
                // and shift the rest on the right of i
                shiftRight(nums, i, i + index);
                // decrement k by fact*index as we can have fact number of
                // permutations for each element at position less than index
                k = k - fact * index;
            }

            return nums;
        }

        public int[] nextEven(final int[] digits) {
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
            // if y doesn’mergedRank contain an even then extend y to left until an even
            // found
            while (!evenFound && y - 1 >= 0 && digits[y - 1] % 2 != 0) {
                y--;
            }

            // input is already the largest permutation
            if (y <= 0) {
                return digits[digits.length - 1] % 2 == 0 ? digits : null;
            }

            // try to extend Y such that y contains an even after swapping X[A] with the
            // Y[rank]
            while (y - 1 >= 0) {
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
                    maxEven = digits[i] % 2 == 0 && (maxEven == -1 || (maxEven != -1 && digits[i] > digits[maxEven]))
                            ? i
                            : maxEven;
                }

                // input is already the largest permutation or need to extend y
                if (maxEven == -1) {
                    y--;
                } else {
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
    }

    class TopK {
        
        public int[] topKFrequent(int[] nums, int k) {
            // assume int[] {number, freq}  
            // minHeap 
            PriorityQueue<int[]> topKHeap = new PriorityQueue<>(k, (e1, e2) -> e1[1] - e2[1]);
            Map<Integer, Integer> freqMap = new HashMap<>();

            // compute freq - O(n)
            for (int num : nums) {
                freqMap.put(num, 1+freqMap.getOrDefault(num, 0));
            }
            // build heap O(nlgk)
            for (int num : freqMap.keySet()) {
                topKHeap.offer(new int[]{num, freqMap.get(num)});
                // remove the least frequent one
                if(topKHeap.size() > k){
                    topKHeap.poll();
                }
            }

            // extract the tops - O(k)
            int[] topK = new int[k];
            int i = 0;
            while (topKHeap.size() > 0) {
                topK[i++] = topKHeap.poll()[0];
            }

            return topK;
        }
        
        public List<String> topKFrequentWords(String[] words, int k) {
            Map<String, Integer> freqMap = new HashMap<>();
            // minHeap 
            // if same freq then lexico shorter comes first
            PriorityQueue<String> topKHeap = new PriorityQueue<>(k, (e1, e2) -> (freqMap.get(e1) == freqMap.get(e2) ? e2.compareTo(e1) : freqMap.get(e1) - freqMap.get(e2)));
            
            // compute freq - O(n)
            for (String w : words) {
                freqMap.put(w, 1+freqMap.getOrDefault(w, 0));
            }
            
            // build heap O(nlgk)
            for (String w : freqMap.keySet()) {
                topKHeap.offer(w);
                // remove the least frequent one
                if(topKHeap.size() > k){
                    topKHeap.poll();
                }
            }

            // extract the tops - O(k)
            List<String> topK = new LinkedList<>();
            while (topKHeap.size() > 0) {
                topK.add(0, topKHeap.poll());
            }

            return topK;
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
        
        public int[][] kClosest(int[][] points, int k) {
            // max heap - reverse order
            final PriorityQueue<int[]> kClosest = new PriorityQueue<>(k, (p1, p2) -> Integer.compare(p2[0]*p2[0] + p2[1]*p2[1], p1[0]*p1[0] + p1[1]*p1[1]));

            for (int i = 0; i < points.length; i++) {
                kClosest.add(points[i]);
                
                // if more than k then rmeove the top/max one
                if (kClosest.size() > k) {
                    kClosest.remove();
                } 
            }

            int[][] res = new int[k][2];
            while (k > 0) {
                res[--k] = kClosest.poll();
            }
            
            return res;
        }

        public Point[] closestk(final Point points[], final int k) {
            // max heap
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
    }

    class Iterators {
        class BSTIterator {
            Stack<TreeNode> stack = new Stack<>();

            public BSTIterator(TreeNode root) {
                while (root != null) {
                    stack.push(root);
                    root = root.left;
                }
            }

            /** @return the next smallest number */
            public int next() {
                if (!hasNext()) {
                    return Integer.MIN_VALUE;
                }

                TreeNode root = stack.pop();
                TreeNode it = root;

                if (it.right != null) {
                    it = it.right;

                    while (it != null) {
                        stack.push(it);
                        it = it.left;
                    }
                }

                return root.val;
            }

            /** @return whether we have a next smallest number */
            public boolean hasNext() {
                return !stack.isEmpty();
            }
        }

        class NestedInteger {
            Integer e;
            List<NestedInteger> list;

            public boolean isInteger() {
                return this.e != null;
            }

            public Integer getInteger() {
                return this.e;
            }

            public List<NestedInteger> getList() {
                return this.list;
            }
        }

        public class NestedIterator implements Iterator<Integer> {
            Stack<NestedInteger> stack;

            public NestedIterator(List<NestedInteger> nestedList) {
                stack = new Stack<>();
                pushToStackInReverseOrder(nestedList);
            }

            @Override
            public Integer next() {
                if (!hasNext())
                    return null;

                return stack.pop().getInteger();
            }

            @Override
            public boolean hasNext() {
                while (!stack.isEmpty() && !stack.peek().isInteger()) {
                    NestedInteger ni = stack.pop();
                    pushToStackInReverseOrder(ni.getList());
                }

                return !stack.isEmpty();
            }

            private void pushToStackInReverseOrder(List<NestedInteger> nestedList) {
                ListIterator<NestedInteger> it = nestedList.listIterator(nestedList.size());
                while (it.hasPrevious()) {
                    stack.push(it.previous());
                }
            }
        }
    }

    class WaterContainer {
        /*
         * Given n non-negative integers representing an elevation map where the width
         * of each bar is 1, compute how much water it can trap after raining. Input:
         * height = [0,1,0,2,1,0,1,3,2,1,2,1] Output: 6 Explanation: The above elevation
         * map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units
         * of rain water are being trapped.
         */
        public int trap(int[] tower) {
            final int n = tower.length;
            if (n == 0) {
                return 0;
            }

            int i = 0, j = n - 1;
            int leftMax = Integer.MIN_VALUE, rightMax = Integer.MIN_VALUE;
            int trappedWater = 0;

            // track max tower on left and on right. Water should be trappaed in between
            // them
            while (i <= j) {
                leftMax = Math.max(leftMax, tower[i]);
                rightMax = Math.max(rightMax, tower[j]);

                // water height will be upto the shorter tower
                if (leftMax < rightMax) {
                    trappedWater += (leftMax - tower[i]);
                    i++;
                } else {
                    trappedWater += (rightMax - tower[j]);
                    j--;
                }
            }

            return trappedWater;
        }

        /*
         * Given n non-negative integers a1, a2, ..., an , where each represents a point
         * at coordinate (i, ai). n vertical lines are drawn such that the two endpoints
         * of the line i is at (i, ai) and (i, 0). Find two lines, which, together with
         * the x-axis forms a container, such that the container contains the most
         * water.
         * 
         * Input: height = [1,8,6,2,5,4,8,3,7] Output: 49 Explanation: The above
         * vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case,
         * the max area of water (blue section) the container can contain is 49.
         */
        public int ContaineWithMostWaterMaxArea(int[] height) {
            int len = height.length, low = 0, high = len - 1;
            int maxArea = 0;
            while (low < high) {
                maxArea = Math.max(maxArea, (high - low) * Math.min(height[low], height[high]));
                if (height[low] < height[high]) {
                    low++;
                } else {
                    high--;
                }
            }
            return maxArea;
        }
    }

    class NumericalComputation {

        public int majorityElement(int[] nums) {
            int count = 1;
            int candidate = nums[0];
            
            for(int i = 1; i < nums.length; i++){
                if(nums[i] == candidate) count++;
                else if (count == 0){
                    candidate = nums[i];
                    count = 1;
                }
                else{
                    count--;
                }
            }
            
            return candidate;
        }
        
        /**
         * Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times.
         * Follow-up: Could you solve the problem in linear time and in O(1) space?
         * @param nums
         * @return
         */
        public List<Integer> majorityElementOneThird(int[] nums) {
            int cand1Count = 0;
            int cand2Count = 0;
            int candidate1 = 0; // choose an number for candidate 1
            int candidate2 = 1; // choose a differenr number for candidatw 2

            for(int i = 0; i < nums.length; i++){
                if(nums[i] == candidate1) cand1Count++;
                else if(nums[i] == candidate2) cand2Count++;
                else if (cand1Count == 0){
                    candidate1 = nums[i];
                    cand1Count = 1;
                }
                else if (cand2Count == 0){
                    candidate2 = nums[i];
                    cand2Count = 1;
                }
                else{
                    cand1Count--;
                    cand2Count--;
                }
            }

            // do 2nd round 
            cand1Count = 0;
            cand2Count = 0;
            for (int i = 0; i < nums.length; i++) {
                if (nums[i] == candidate1)
                    cand1Count++;
                else if (nums[i] == candidate2)
                    cand2Count++;
            }
            
            List<Integer> res = new ArrayList<>();
            if(cand1Count > nums.length/3){
                res.add(candidate1);
            }
            if(cand2Count > nums.length/3){
                res.add(candidate2);
            }
            return res;
        }
        
        public double myPow(double x, int n) {
            if (n == 0)
                return 1;
            else if (n > 0)
                return powRec(x, n);
            else
                return 1 / powRec(x, n);
        }

        public double powRec(double x, int y) {
            if (y == 0) {
                return 1.0;
            }
            if (y == 1) {
                return x;
            }

            double pow = powRec(x, y / 2);
            if ((y & 1) != 0) {
                return pow * pow * x;
            } else {
                return pow * pow;
            }
        }

        public int reverseInt(int x) {
            int result = 0;

            while (x != 0) {
                int tail = x % 10;
                int newResult = result * 10 + tail;
                if ((newResult - tail) / 10 != result) {
                    return 0;
                }
                result = newResult;
                x = x / 10;
            }

            return result;
        }

        public int mySqrt(int x) {
            if (x == 0) {
                return 0;
            }
            if (x <= 3) {
                return 1;
            }
            if (x == 4) {
                return 2;
            }

            int l = 1, h = x / 2;
            int root = 0;

            while (l < h) {
                root = l + (h - l) / 2;

                if (root == x / root) {
                    return root;
                } else if (root < x / root) {
                    if ((root + 1) > x / (root + 1)) {
                        return root;
                    }
                    l = root + 1;
                } else if (root > x / root) {
                    if ((root - 1) < x / (root - 1)) {
                        return root - 1;
                    }
                    h = root - 1;
                }
            }

            return l;
        }

        public int myAtoi(String str) {
            int index = 0, sign = 1, total = 0;
            // 1. Empty string
            if (str.length() == 0)
                return 0;

            // 2. Remove Spaces
            while (index < str.length() && str.charAt(index) == ' ')
                index++;
            if (index == str.length())
                return 0;

            // 3. Handle signs
            if (str.charAt(index) == '+' || str.charAt(index) == '-') {
                sign = str.charAt(index) == '+' ? 1 : -1;
                index++;
            }

            // 4. Convert number and avoid overflow
            while (index < str.length()) {
                int digit = str.charAt(index) - '0';
                if (digit < 0 || digit > 9)
                    break;

                // check if total will be overflow after 10 times and add digit
                if (Integer.MAX_VALUE / 10 < total || Integer.MAX_VALUE / 10 == total && Integer.MAX_VALUE % 10 < digit)
                    return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;

                total = 10 * total + digit;
                index++;
            }
            return total * sign;
        }

        public int binaryToDecimal(String str) {
            int dec = 0;

            int b = 1; // 2^0
            for (int i = str.length() - 1; i >= 0; i--) {
                dec += b * (str.charAt(i) - '0');
                b *= 2;
            }

            return dec;
        }

        public int divide(int dividend, int divisor) {
            if (dividend == 0) {
                return 0;
            }
            if (dividend == Integer.MIN_VALUE && divisor == -1) {
                return Integer.MAX_VALUE;
            }
            if (divisor == 1 || divisor == -1) {
                return dividend * divisor;
            }

            int sign = (((dividend > 0 && divisor > 0) || (dividend < 0 && divisor < 0)) ? 1 : -1);
            long dividendLong = Math.abs((long) dividend);
            long divisorLong = Math.abs((long) divisor);
            int res = 0;

            while (dividendLong >= divisorLong) {
                dividendLong -= divisorLong;
                res++;
            }

            if (sign == -1) {
                return -res;
            } else {
                return res;
            }
        }

        /**
         * Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be
         * validated according to the following rules: 1. Each row must contain the
         * digits 1-9 without repetition. 2. Each column must contain the digits 1-9
         * without repetition. 3. Each of the nine 3 x 3 sub-boxes of the grid must
         * contain the digits 1-9 without repetition.
         * 
         * @param board
         * @return
         */
        public boolean isValidSudoku(char[][] board) {
            for (int i = 0; i < 9; i++) {
                HashSet<Character> rows = new HashSet<Character>();
                HashSet<Character> columns = new HashSet<Character>();
                HashSet<Character> cube = new HashSet<Character>();
                for (int j = 0; j < 9; j++) {
                    if (board[i][j] != '.' && !rows.add(board[i][j]))
                        return false;
                    if (board[j][i] != '.' && !columns.add(board[j][i]))
                        return false;
                    int RowIndex = 3 * (i / 3);
                    int ColIndex = 3 * (i % 3);
                    if (board[RowIndex + j / 3][ColIndex + j % 3] != '.'
                            && !cube.add(board[RowIndex + j / 3][ColIndex + j % 3]))
                        return false;
                }
            }
            return true;
        }
        
        class FindTwoMissing {
            public void find2Missing(int[] a, int n){
                for(int i = 0; i < a.length; i++){
                    if(a[Math.abs(a[i])-1] > 0){
                        a[Math.abs(a[i])-1]  = -a[Math.abs(a[i])-1];
                    }
                }
                
                for(int i = 0; i < a.length; i++){
                    if(a[i] > 0){
                        System.out.println("missing: "+i+1);
                    }
                    else{
                        a[i] = -a[i];
                    }
                }
            }
            
            //O(n) time, O(1) space
            public void findMissing2(int a[], int n){
                int mask = 0;
                
                //O(n)
                for(int i = 1; i<=n; i++){
                    mask ^= i;
                }
                
                //O(n)
                for(int i = 0; i < a.length; i++){
                    mask ^= a[i];
                }
                
                //get the right most set bit
                mask &= ~(mask-1);
                int mis1=0, mis2=0;
                for(int i = 0; i<a.length; i++){
                    if((a[i]&mask) == mask){
                        mis1 ^= a[i];
                    }
                    else{
                        mis2 ^= a[i];
                    }
                }
                
                for(int i = 1; i<=n; i++){
                    if((i&mask) == mask){
                        mis1 ^= i;
                    }
                    else{
                        mis2 ^= i;
                    }
                }
                
                System.out.println("missing numbers : "+mis1+", "+mis2);
            }
        }
        
        /**
         * Given an unsorted integer array nums, find the smallest missing positive integer.
         * 
         * Example 1:
            
            Input: nums = [1,2,0]
            Output: 3
            
            Example 2:
            
            Input: nums = [3,4,-1,1]
            Output: 2
            
            Example 3:
            
            Input: nums = [7,8,9,11,12]
            Output: 1

         * @param nums
         * @return
         */
        public int firstMissingPositive(int[] nums) {
            if (nums.length == 0) {
                return 1;
            }

            int p = 0;
            int r = nums.length - 1;
            int q = p - 1;

            // if there are negatives move them at the end
            // partition positive and negatives. Left partition is all positives
            // and right partition is all negative or zero
            for (int j = 0; j <= r; j++) {
                if (nums[j] > 0) {
                    swap(nums, ++q, j);
                }
            }

            q++;
            // now go through positive numbers and keep a map per positive number to check
            // if exists
            // we can reuse the array to avoid extra space
            for (int i = 0; i < q; i++) {
                // use the number itself as the positional index in the array
                int index = Math.abs(nums[i]) - 1;
                // ideally each number should appear in the index at number - 1
                // if anything is missing than they wouldn't match
                // mark the numbers that are in their own position. We can negate the number as
                // a marker.
                // so any position not holding their own number will not contain negative value
                if (index < q) {
                    nums[index] = -Math.abs(nums[index]);
                }
            }

            // first position with positive number is the desired first missing positive
            for (int i = 0; i < q; i++) {
                if (nums[i] > 0) {
                    return i + 1;
                }
            }

            return q + 1;
        }

        public String countAndSay(int n) {
            StringBuilder curr = new StringBuilder("1");
            StringBuilder prev;
            int count;
            char say;
            for (int i = 1; i < n; i++) {
                prev = curr;
                curr = new StringBuilder();
                count = 1;
                say = prev.charAt(0);

                for (int j = 1, len = prev.length(); j < len; j++) {
                    if (prev.charAt(j) != say) {
                        curr.append(count).append(say);
                        count = 1;
                        say = prev.charAt(j);
                    } else
                        count++;
                }
                curr.append(count).append(say);
            }
            return curr.toString();
        }
        
        /**
         *     Given two big integers represented as strings, Multiplication them 
         *     and return the production as string.
         *     
         *     For example, given a=2343324 and b=232232 then 
         *     return c = a*b = 23433242334323342 * 23223233232434324 = 544195652122144709711313995190808
         * @param str1
         * @param str2
         * @return
         */
        public String multiplyBigIntegers(String str1, String str2){
            String res = new String("0");
            
            int count = 0;
            for(int i = str2.length()-1; i>=0 ; i--){
                int d2 = str2.charAt(i)-'0';
                
                int carry = 0;
                StringBuffer prod = new StringBuffer();
                for(int j = str1.length()-1; j>=0; j--){
                    int d1 = str1.charAt(j)-'0';
                    int p = carry+(d1*d2);
                    prod.append(p%10);
                    carry = p/10;
                }
                
                if(carry != 0){
                    prod.append(carry);
                }
                
                prod.reverse();

                for(int k = 0; k<count; k++){
                    prod.append(0);
                }
                
                res = add(res, prod.toString());
                count++;
            }
            
            return res.toString();
        }

        //O(n);
        private String add(String str1, String str2){
            StringBuffer res = new StringBuffer();
            
            int i = str1.length()-1;
            int j = str2.length()-1;
            int carry = 0;
            while(true){
                if(i < 0 && j < 0){
                    break;
                }
                
                int d1 = i < 0 ? 0 : str1.charAt(i--)-'0';
                int d2 = j < 0 ? 0 : str2.charAt(j--)-'0';
                int sum = d1+d2+carry;
                
                res.append(sum%10);
                carry = sum/10;
            }
            
            if(carry != 0){
                res.append(carry);
            }
            
            return res.reverse().toString();
        }

        /*
         * You are given two non-empty linked lists representing two non-negative
         * integers. The digits are stored in reverse order, and each of their nodes
         * contains a single digit. Add the two numbers and return the sum as a linked
         * list.
         */
        public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
            int carryover = 0;
            ListNode result = null;
            ListNode headResult = new ListNode(-1);// dummy
            while (l1 != null || l2 != null || carryover > 0) {
                int first = (l1 != null) ? l1.val : 0;
                int second = (l2 != null) ? l2.val : 0;
                int sum = (first + second + carryover) % 10;
                carryover = (first + second + carryover) / 10;

                if (result == null) {
                    result = new ListNode(sum);
                    headResult.next = result;
                } else {
                    result.next = new ListNode(sum);
                    result = result.next;
                }

                l1 = (l1 == null) ? l1 : l1.next;
                l2 = (l2 == null) ? l2 : l2.next;
            }

            return headResult.next;
        }

        /*
         * You are given two non-empty linked lists representing two non-negative
         * integers. The most significant digit comes first and each of their nodes
         * contain a single digit. Add the two numbers and return it as a linked list.
         * 
         * Example: Input: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4 Output: 7 -> 8 -> 0 -> 7
         */
        public ListNode addTwoNumbersMSB(ListNode l1, ListNode l2) {
            // two stack
            // no modifications of input lists
            Stack<Integer> st1 = new Stack<>();
            Stack<Integer> st2 = new Stack<>();

            while (l1 != null) {
                st1.push(l1.val);
                l1 = l1.next;
            }
            while (l2 != null) {
                st2.push(l2.val);
                l2 = l2.next;
            }

            int sum = 0;
            int carry = 0;
            ListNode dummyHead = new ListNode(-1);
            while (!st1.isEmpty() || !st2.isEmpty() || carry != 0) {
                sum = carry;
                if (!st1.isEmpty()) {
                    sum += st1.pop();
                }
                if (!st2.isEmpty()) {
                    sum += st2.pop();
                }

                ListNode node = new ListNode(sum % 10);
                node.next = dummyHead.next;
                dummyHead.next = node;

                carry = sum / 10;
            }

            return dummyHead.next;
        }

        public int fourListSum(int A[], int B[], int C[], int D[]) {
            Map<Integer, Integer> abCount = new HashMap<>();
            Arrays.stream(A).forEach(a -> {
                Arrays.stream(B).forEach(b -> {
                    abCount.put(a + b, 1 + abCount.getOrDefault(a + b, 0));
                });
            });

            AtomicInteger counter = new AtomicInteger(0);
            Arrays.stream(C).forEach(c -> {
                Arrays.stream(D).forEach(d -> {
                    counter.addAndGet(abCount.getOrDefault(-(c + d), 0));
                });
            });

            return counter.get();
        }

        public void validIp(List<String> logs) {
            String validIp = "(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])";
            String pattern = validIp + "\\." + validIp + "\\." + validIp + "\\." + validIp;
            Pattern p = Pattern.compile(pattern);
            Set<String> ips = new HashSet<>();
            Matcher m;

            for (String log : logs) {
                m = p.matcher(log);

                while (m.find()) {
                    String ip = m.group().trim();
                    if (!ips.contains(ip)) {
                        ips.add(ip);
                        System.out.println(ip);
                    }
                }
            }

        }

        public boolean isCrossed(double[] s) {
            // base case
            if (s.length < 4) {
                return false;
            }
            if (s[0] >= s[2] && s[3] >= s[1]) {
                return true;
            }

            // test if the moves are on outward increasing spiral
            int i = 3;
            while (i < s.length) {
                if (s[i] > s[i - 2] && s[i - 1] > s[i - 3])
                    i++;
                else
                    break;
            }

            // if we visited all the moves then there is no intersection
            if (i == s.length) {
                return false;
            }

            // otherwise moves are on A decreasing inward spiral starting from i
            // we first need check if the two spirals are crossing each other which can only
            // possible
            // when edge i+1 crosses edge (i-4) or edge i+1 crosses edge i-2 (if exists)

            if (i < s.length && i > 3 && s[i + 1] >= (s[i - 1] - s[i - 3])) {
                if (s[i] >= (s[i - 2] - s[i - 4]) || s[i + 1] >= s[i - 1])
                    return true;
            }

            // if two spiral didn'mergedRank intersect then check for decreasing s
            while (i + 3 < s.length) {
                if (s[i] > s[i + 2] && s[i + 1] > s[i + 3]) {
                    i++;
                } else
                    break;
            }

            // if we visited all the moves then there is no intersection
            if (i + 3 == s.length) {
                return false;
            }

            return false;
        }

        /**
         *  Given a number “n”, find the least number of perfect square numbers that sum to n

            For Example:
            n=12, return 3 (4 + 4 + 4) = (2^2 + 2^2 + 2^2) NOT (3^2 + 1 + 1 + 1)
            n = 6, return 3 (4 + 1 + 1) = (2^2 + 1^2 + 1^2)
            
         * @param n
         * @return
         */
        public int perfectSquareDP(int n) {
            if (n <= 0) {
                return 0;
            }

            int[] dp = new int[n + 1];
            Arrays.fill(dp, Integer.MAX_VALUE);
            dp[0] = 0;
            dp[1] = 1;

            // to compute least perfect for n we compute top down for each
            // possible value sum from 2 to n
            for (int i = 2; i <= n; i++) {
                // for a particular value i we can break it as sum of a perfect square j*j and
                // all perfect squares from solution of the remainder (i-j*j)
                for (int j = 1; j * j <= i; j++) {
                    dp[i] = Math.min(dp[i], 1 + dp[i - j * j]);
                }
            }

            return dp[n];
        }

        private boolean is_square(int n) {
            int sqrt_n = (int) (Math.sqrt(n));
            return (sqrt_n * sqrt_n == n);
        }

        // Based on Lagrange's Four Square theorem, there
        // are only 4 possible results: 1, 2, 3, 4.
        public int perfectSquaresLagrange(int n) {
            // If n is a perfect square, return 1.
            if (is_square(n)) {
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
            int sqrt_n = (int) (Math.sqrt(n));
            for (int i = 1; i <= sqrt_n; i++) {
                if (is_square(n - i * i)) {
                    return 2;
                }
            }

            return 3;
        }

        public void rotateRight(int[] A, int k) {
            int n = A.length;
            if (n <= 1) {
                return;
            }

            k = k % n;

            if (k == 0) {
                return;
            }

            // reverse non rotated part
            reverse(A, 0, n - k - 1);
            // reverse rotated part
            reverse(A, n - k, n - 1);
            // reverse the whole array
            reverse(A, 0, n - 1);
        }

        public void rotateLeft(int[] A, int k) {
            int n = A.length;
            if (n <= 1) {
                return;
            }

            k = k % n;

            if (k == 0) {
                return;
            }

            // reverse the whole array
            reverse(A, 0, n - 1);
            // reverse rotated part
            reverse(A, n - k, n - 1);
            // reverse non rotated part
            reverse(A, 0, n - k - 1);
        }

        public int nthUglyNumber(int n) {
            int nthUgly = 1;
            PriorityQueue<Integer> minHeap = new PriorityQueue<Integer>();
            Set<Integer> uniques = new HashSet<Integer>();
            minHeap.offer(1);

            while (n > 0) {
                nthUgly = minHeap.poll();
                int next = nthUgly * 2;
                if (nthUgly <= Integer.MAX_VALUE / 2 && !uniques.contains(next)) {
                    minHeap.offer(next);
                    uniques.add(next);
                }
                next = nthUgly * 3;
                if (nthUgly <= Integer.MAX_VALUE / 3 && !uniques.contains(next)) {
                    minHeap.offer(next);
                    uniques.add(next);
                }
                next = nthUgly * 5;
                if (nthUgly <= Integer.MAX_VALUE / 5 && !uniques.contains(next)) {
                    minHeap.offer(next);
                    uniques.add(next);
                }
                n--;
            }

            return nthUgly;
        }

        public int nthUglyDP(int n) {
            int merged[] = new int[n];
            // 1 is considered as ugly so, its the first ugly number
            merged[0] = 1;
            // pointer to the three sets of ugly numbers generated by multiplying
            // respectively by 2, 3, and 5
            // p2 points to current ugly of the sequence : 1*2, 2*2, 3*2, 4*2, ...
            // p3 points to current ugly of the sequence : 1*3, 2*3, 3*3, 4*3, ...
            // p5 points to current ugly of the sequence : 1*5, 2*5, 3*5, 4*5, ...
            int p2 = 0, p3 = 0, p5 = 0;

            // merge the 3 sequences pointed by p2, p3, and p5 and always take the min as we
            // do in merge sort
            for (int i = 1; i < n; i++) {
                merged[i] = Math.min(Math.min(merged[p2] * 2, merged[p3] * 3), merged[p5] * 5);

                // now increment the corrsponding pointer - same number can be generated in
                // multiple sequences
                // for example, 10 can be genetaed by 2 as 5*2 or by 5 as 2*5. So, we increment
                // all pointers
                // that contains same value to avoid duplicates
                if (merged[i] == merged[p2] * 2) {
                    p2++;
                }
                if (merged[i] == merged[p3] * 3) {
                    p3++;
                }
                if (merged[i] == merged[p5] * 5) {
                    p5++;
                }
            }

            return merged[n - 1];
        }

        /**
         *  Given a unsorted array with n elements. How can we find the largest gap 
         *  between consecutive numbers of sorted version in O(n)? 

            For example, we have a unsorted array, a=[5, 3, 1, 8, 9, 2, 4] of size n=7 then the sorted version is
            [1, 2, 3, 4, 5, 8, 9]. The output of the algorithm should be 8-5 = 3.
            
            Similarly, for a=[5, 1, 8, 9, 999999, 99999] then answer should be 999999-99999 = 900000. 
         * @param a
         * @return
         */
        public int maxGap(int[] a) {
            int n = a.length;
            if (n < 2) {
                return 0;
            }

            int max = Integer.MIN_VALUE;
            int min = Integer.MAX_VALUE;

            for (int i = 0; i < n; i++) {
                max = Math.max(max, a[i]);
                min = Math.min(min, a[i]);
            }

            // n-1 buckets - we only care about max and min in each buckets
            int[] bucketMaxima = new int[n - 1];
            Arrays.fill(bucketMaxima, Integer.MIN_VALUE);
            int[] bucketMinima = new int[n - 1];
            Arrays.fill(bucketMinima, Integer.MAX_VALUE);
            // bucket width
            float delta = (float) (max - min) / ((float) n - 1);

            // populate the bucket maxima and minima
            for (int i = 0; i < n; i++) {
                if (a[i] == max || a[i] == min) {
                    continue;
                }

                int bucketIndex = (int) Math.floor((a[i] - min) / delta);
                bucketMaxima[bucketIndex] = bucketMaxima[bucketIndex] == Integer.MIN_VALUE ? a[i]
                        : Math.max(bucketMaxima[bucketIndex], a[i]);
                bucketMinima[bucketIndex] = bucketMinima[bucketIndex] == Integer.MAX_VALUE ? a[i]
                        : Math.min(bucketMinima[bucketIndex], a[i]);
            }

            // find the maxgap - maxgaps
            int prev = min;
            int maxGap = 0;
            for (int i = 0; i < n - 1; i++) {
                // empty bucket according to Pigeonhole principle
                if (bucketMinima[i] == Integer.MAX_VALUE) {
                    continue;
                }

                maxGap = Math.max(maxGap, bucketMinima[i] - prev);
                prev = bucketMaxima[i];
            }

            maxGap = Math.max(maxGap, max - prev);

            return maxGap;
        }
        
        /**
         *     Given a set S of digits [0-9] and a number n. Find the smallest integer larger than n (ceiling) 
         *     using only digits from the given set S. You can use a value as many times you want.
         *     
         *     For example, d=[1, 2, 4, 8] and n=8753 then return 8811. 
         *     For, d=[0, 1, 8, 3] and n=8821 then return 8830. 
         *     For d=[0, 1, 8, 3] and n=8310 then return 8311.
         * @param digits
         * @param n
         * @return
         */
        public int[] nextHigherWithDigits(int[] digits, int n){
            //get the target digits sorted
            int[] sortedDigits = Arrays.copyOf(digits, digits.length);
            Arrays.sort(sortedDigits);
            
            //get the digits of the number from LSB to MSB oder
            ArrayList<Integer> nums = new ArrayList<Integer>();
            while(n>0){
                nums.add(n%10);
                n/=10;
            }
            
            //reverse to get the digits in MSB to LSB order
            Collections.reverse(nums);
            
            boolean higherAdded = false;
            int[] res = new int[nums.size()];
            int i = 0;
            //for each digit in thr number find the next higher in the sorted target digits
            for(int num : nums){
                //if a higher digit was already found in previous step then rest of the digits should have the smallest digit
                if(higherAdded){
                    //add the smallest digit
                    res[i++] = sortedDigits[0];
                    continue;
                }
                
                //otherwise , find the next higher (or equal) digit
                int nextHigher = binarySearchCeiling(sortedDigits, 0, sortedDigits.length-1, num);
                //if no such higher digit then no solution
                if(nextHigher == -1){
                    return null;
                }
                //otherwise if the digit is indeed higher then all subsequent digits should be smallest, so mark this event 
                else if(sortedDigits[nextHigher] > num){
                    higherAdded = true;
                }
                
                //add the next higher (or equal digit)
                res[i++] = sortedDigits[nextHigher];
            }
            
            //If we didn;t find any higher digit, which is only possible when we found all equal digits
            //then set the LSB to the next strictly higher number (not equal)
            if(!higherAdded){
                int nextHigher = binarySearchCeiling(sortedDigits, 0, sortedDigits.length-1, res[i-1]+1);
                if(nextHigher == -1){
                    return null;
                }
                
                res[i-1] = sortedDigits[nextHigher];
            }
            
            return res;
        }
        
        public int binarySearchCeiling(int A[], int l, int h, int key){
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
        
        /**
         *  Given a string, rearrange characters of the string such that no duplicate characters are 
         *  adjacent to each other. 
            For example,
            
            Input: aaabc
            Output: abaca
            
            Input: aa
            Output: No valid output
            
            Input: aaaabc
            Output: No valid output
         * @param str
         * @return
         */
        public String rearrangeAdjacentDuplicates(String str){
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

    }

    class Parenthesis {
        // n = pairs of parentheis
        // complexity = O((4^n)/√n) -- nth catalan number
        public List<String> generateParenthesis(int n) {
            ArrayList<String> res = new ArrayList<String>();
            if (n <= 0) {
                return res;
            }

            generateParenthesisBackTrack("", 0, 0, n, res);

            return res;
        }

        /**
         * Instead of adding '(' or ')' every time, let's only add them when we know it
         * will remain a valid sequence. We can do this by keeping track of the number
         * of opening and closing brackets we have placed so far.
         * 
         * We can start an opening bracket if we still have one (of n) left to place.
         * And we can start a closing bracket if it would not exceed the number of
         * opening brackets.
         */
        public void generateParenthesisBackTrack(String cur, int open, int close, int max, List<String> list) {

            if (cur.length() == max * 2) {
                list.add(cur);
                return;
            }

            // we can add an opening parenthesis as long as we haven't use all of them
            if (open < max)
                generateParenthesisBackTrack(cur + "(", open + 1, close, max, list);
            // we can add a closing parenthesis as long as we have enough matching opening
            // parenthesis
            if (close < open)
                generateParenthesisBackTrack(cur + ")", open, close + 1, max, list);
        }

        public int longestValidParentheses(String s) {
            if (s == null || s.length() == 0) {
                return 0;
            }
            // Motivation: a string becomes invalid for the first unmatched ‘)’ or the last
            // unmatched ‘(‘.
            // for first unmatched ')' we can do forward pass
            // for last unmatched '(' we can do backward pass
            int curLen = 0;
            int maxLen = 0;
            int count = 0;
            // forward pass
            for (int i = 0; i < s.length(); i++) {
                // count as log as we encounter '(' till we see a ')'
                if (s.charAt(i) == '(') {
                    count++; // equivalent to stack push
                }
                // if we encounter a ')' then we decrease counter
                else {
                    // we decrease counter (aka pop from stack) only if a matching '(' was seen
                    // before. Otherwise we found an invalid ')' . So reset the substring
                    if (count <= 0) {
                        curLen = 0;
                    }
                    // decrease (pop)
                    else {
                        count--;
                        curLen += 2;

                        // if matching '(' was found and no more opening bracket (count = 0) //in the
                        // stack then update max
                        if (count == 0) {
                            maxLen = Math.max(maxLen, curLen);
                        }
                    }
                }
            }

            // backward pass
            curLen = 0;
            count = 0;
            for (int i = s.length() - 1; i >= 0; i--) {
                // count as log as we encounter ')' till we see a '('
                if (s.charAt(i) == ')') {
                    count++; // equivalent to stack push
                }
                // if we encounter a '(' then we decrease counter
                else {
                    // we decrease counter (aka pop from stack) only if a matching ')' was seen
                    // before. Otherwise we found an invalid '(' . So reset the substring
                    if (count <= 0) {
                        curLen = 0;
                    }
                    // decrease (pop)
                    else {
                        count--;
                        curLen += 2;

                        // if matching '(' was found and no more opening bracket (count = 0) //in the
                        // stack then update max
                        if (count == 0) {
                            maxLen = Math.max(maxLen, curLen);
                        }
                    }
                }
            }

            return maxLen;
        }
    }

    class DP {
        public int climbStairs(int n) {
            // it's basically fibbonacy starting from 1
            // 1, 1, 2, 3, 5, 8, ..
            if (n <= 1) {
                return 1;
            }

            // return climbStairs(n-1) + climbStairs(n-2);
            // do with DP
            int[] dp = new int[n + 1];
            dp[0] = 1;
            dp[1] = 1;
            for (int i = 2; i <= n; i++) {
                dp[i] = dp[i - 1] + dp[i - 2];
            }

            return dp[n];
        }
        
        public boolean canJump(int[] nums) {
            int n = nums.length;
            boolean dp[] = new boolean[n];
            dp[n-1] = true;
            
            for(int i = n-2; i>=0; i--){
                int j = i+1;
                int k = Math.min(j+nums[i], n);
                while(j < k){
                    if(dp[j++]){
                        dp[i] = true;
                        break;
                    }
                }
            }
            
            return dp[0];
        }
        
        /**
         * Given an array of non-negative integers nums, you are initially positioned at the first index of the array.
         * Each element in the array represents your maximum jump length at that position.
         * Your goal is to reach the last index in the minimum number of jumps.
         * 
         * Input: nums = [2,3,1,1,4]
         * Output: 2
         * Explanation: The minimum number of jumps to reach the last index is 2. 
         * Jump 1 step from index 0 to 1, then 3 steps to the last index.
         * 
         * Input: nums = [2,3,0,1,4]
         * Output: 2
         * 
         * @param nums
         * @return
         */
        public int minNumberOfJumps(int[] nums) {
            int n = nums.length;
            long dp[] = new long[n];
            Arrays.fill(dp, Integer.MAX_VALUE);
            dp[n-1] = 0;

            for(int i = n-2; i>=0; i--){
                int j = i+1;
                int k = Math.min(j+nums[i], n);
                while(j < k){
                    // if
                    if(j != n-1 && nums[j] == 0){
                        j++;
                        continue;
                    }
                    // we can reach j from i by one jump
                    if((dp[j]+1) < dp[i]){
                        dp[i] = dp[j]+1;
                    }
                    j++;
                }
            }

            return (int) dp[0];
        }
        
        public int[] slidingWindowMax(final int[] in, final int w) {
            if(in.length == 0 || w == 0){
                return in;
            }
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
        
        class MinsumPathTriangle {
            /**
             *  Given a triangle, find the minimum path sum from top to bottom. 
             *  Each step you may move to adjacent numbers on the row below.

                For example, given the following triangle
                
                     [2],
                    [3,4],
                   [6,5,7],
                  [4,1,8,3]
                
                The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).
                
                But for the following triangle –
                
                     [2],
                    [5,4],
                   [5,5,7],
                  [1,4,8,3]
                
                The minimum path sum from top to bottom is 11 (i.e., 2 + 5 + 5 + 1 = 13).
             * @param triangle
             * @return
             */
            //O(n^2) time and O(n^2) space for dp table
            public int triangleMinSumPath(List<int[]> triangle){
                /**
                 * At each level we need to choose the node that yields a min total sum with the following relation –

                   dp[level][i] = triangle[level][i] + min{dp[next_level][i], dp[next_level][i+1]}
                 */
                int levels = triangle.size();
                int dp[][] = new int[levels][levels];
                
                dp[levels-1] = triangle.get(levels-1);
                
                //bottom up Dijkstra
                for(int l = levels-2; l>=0 ; l--){
                    for(int i = 0; i<=l; i++){
                        dp[l][i] = Math.min(dp[l+1][i], dp[l+1][i+1]) + triangle.get(l)[i];
                    }
                }
                return dp[0][0];
            }
            
            //O(n^2) time and O(n) space 
            public int triangleMinSumPath2(List<int[]> triangle){
                /**
                 * If we print the dp table of the above code for example 2 then we will see the following –
                    Triangle - 
                    
                         [2],
                        [5,4],
                       [5,5,7],
                      [1,4,8,3]
                    
                    
                    dp table  -
                    
                    13,  0,  0, 0 
                    11, 13,  0, 0 
                     6,  9, 10, 0 
                     1,  4,  8, 3 
                     
                    If we look closely then we can see that the table has meaningful values in lower half 
                    only and at each level bottom up we have one of the column value getting fixed. 
                    So, we could have basically used the bottom level array as the dp table and at each 
                    level we update the columns bottom up.

                 */
                int levels = triangle.size();
                int dp[] = new int[levels];
                
                dp = triangle.get(levels-1);
                
                //bottom up Dijkstra
                for(int l = levels-2; l>=0 ; l--){
                    for(int i = 0; i<=l; i++){
                        dp[i] = Math.min(dp[i], dp[i+1]) + triangle.get(l)[i];
                    }
                }
                return dp[0];
            }
        }
    }

    class DPGrid {
        /**
         * Given a 2D binary matrix filled with 0's and 1's, find the largest square
         * containing only 1's and return its area.
         */
        public int maximalSquare(char[][] a) {
            if (a.length == 0)
                return 0;
            int m = a.length, n = a[0].length, result = 0;
            int[][] dp = new int[m + 1][n + 1];
            for (int i = 1; i <= m; i++) {
                for (int j = 1; j <= n; j++) {
                    if (a[i - 1][j - 1] == '1') {
                        dp[i][j] = Math.min(Math.min(dp[i][j - 1], dp[i - 1][j - 1]), dp[i - 1][j]) + 1;
                        result = Math.max(dp[i][j], result); // update result
                    }
                }
            }
            return result * result;
        }

        public int minPathSum(int[][] grid) {
            if (grid == null || grid.length == 0) {
                return 0;
            }

            int dp[][] = new int[grid.length][grid[0].length];
            dp[0][0] = grid[0][0];
            // if only go down then sum is increasing
            for (int i = 1; i < grid.length; i++) {
                dp[i][0] = dp[i - 1][0] + grid[i][0];
            }
            // if only go right then sum is increasing
            for (int j = 1; j < grid[0].length; j++) {
                dp[0][j] = dp[0][j - 1] + grid[0][j];
            }

            // now walk - we can reach a grid either from top (i-1, j) or from left (i, j-1)
            for (int i = 1; i < grid.length; i++) {
                for (int j = 1; j < grid[0].length; j++) {
                    dp[i][j] = grid[i][j] + Math.min(dp[i - 1][j], dp[i][j - 1]);
                }
            }

            return dp[grid.length - 1][grid[0].length - 1];
        }

        public int minPathSum2(int[][] grid) {
            if (grid == null || grid.length == 0) {
                return 0;
            }

            int dp[] = new int[grid[0].length];
            dp[0] = grid[0][0];
            // if only go right then sum is increasing
            for (int j = 1; j < grid[0].length; j++) {
                dp[j] = dp[j - 1] + grid[0][j];
            }

            // now walk - we can reach a grid either from top (i-1, j) or from left (i, j-1)
            for (int i = 1; i < grid.length; i++) {
                dp[0] = dp[0] + grid[i][0];
                for (int j = 1; j < grid[0].length; j++) {
                    dp[j] = grid[i][j] + Math.min(dp[j], dp[j - 1]);
                }
            }

            return dp[grid[0].length - 1];
        }

        public int calculateMinimumHP(int[][] dungeon) {
            if (dungeon == null || dungeon.length == 0) {
                return 0;
            }
            int m = dungeon.length;
            int n = dungeon[0].length;

            // dp[i][j] is minimum health needed at location i,j
            // goal is to minimize dp[0,0] such that knight is alive (>0)
            // we can compute the table bottom up
            int dp[][] = new int[dungeon.length][dungeon[0].length];

            // to be in the princess celll kinght needs at least 1 health after fighting
            // demons
            dp[m - 1][n - 1] = Math.max(1, 1 - dungeon[m - 1][n - 1]);
            // intializie the boudnary conditions bottom up
            // imagine that knight had a oracle and knows all the grids state
            // so, he imagined himslef to start in precess cell and moving up/right with
            // the princess and tracing his route back to initial posiotion

            // moving up along the boundary - at every cell kinght needs at least of 1
            // health after fighting demons
            for (int i = m - 2; i >= 0; i--) {
                dp[i][n - 1] = Math.max(1, dp[i + 1][n - 1] - dungeon[i][n - 1]);
            }
            // moving left along the boundary - at every cell kinght needs at least of 1
            // health after fighting demons
            for (int j = n - 2; j >= 0; j--) {
                dp[m - 1][j] = Math.max(1, dp[m - 1][j + 1] - dungeon[m - 1][j]);
            }

            // now walk bottom up
            for (int i = m - 2; i >= 0; i--) {
                for (int j = n - 2; j >= 0; j--) {
                    int healthUp = Math.max(1, dp[i + 1][j] - dungeon[i][j]);
                    int healthLeft = Math.max(1, dp[i][j + 1] - dungeon[i][j]);
                    // minimize the positive health neeed.
                    dp[i][j] = Math.min(healthUp, healthLeft);
                }
            }

            return dp[0][0];
        }
    }

    class SubArraySumProduct {

        public int maxSumSubArray(int[] a) {
            int localMaxima = a[0];
            int globalMaxima = a[0];

            // we compute localMaxima of subarray so far ending at each index
            for (int i = 1; i < a.length; i++) {
                // for each element we get maximum sum subarray either by extending the
                // localMaxima (prev best subaray)
                // by adding the current element to it, OR we reset and start a new subarray
                // from this element because
                // it yelds largest sum
                localMaxima = Math.max(a[i], localMaxima + a[i]);
                // compute the global maxima across all such subarrays
                globalMaxima = Math.max(globalMaxima, localMaxima);
            }

            return globalMaxima;
        }

        public int maxProductSubarray(int[] a) {
            if (a == null || a.length == 0) {
                return 0;
            }

            // kadanes algorithm
            // at a given element we can have two extreme products
            // one is the most positive value if a[i] is positive (max)
            // other is the most negative if a[i] is negative (min)
            // so we track two extreme max
            int localPositiverMax = a[0];
            int localNegativerMax = a[0];
            int globalMaxProd = a[0];

            for (int i = 1; i < a.length; i++) {
                // if current element is negative then multiplying we can swap localPositiverMax
                // and localNegativerMax
                if (a[i] < 0) {
                    int temp = localPositiverMax;
                    localPositiverMax = localNegativerMax;
                    localNegativerMax = temp;
                }

                // at a given index we make highest product either
                // (1) by multiplying current value with the previous local maxima subarray
                // (2) we start a new subarray at current location by reseting max with the
                // current value
                localPositiverMax = Math.max(a[i], a[i] * localPositiverMax);
                localNegativerMax = Math.min(a[i], a[i] * localNegativerMax);
                // compute the global max so far
                globalMaxProd = Math.max(globalMaxProd, localPositiverMax);
            }

            return globalMaxProd;
        }

        public int subarraySum(int[] nums, int k) {
            int sum = 0;
            int count = 0;
            // map from sum to count of such prefixes
            final Map<Integer, Integer> candidates = new HashMap<>();
            candidates.put(0, 1);

            for (int i = 0; i < nums.length; i++) {
                sum += nums[i];

                if (candidates.containsKey(sum - k)) {
                    // a subarray found
                    count += candidates.get(sum - k);
                }

                candidates.put(sum, candidates.getOrDefault(sum, 0) + 1);
            }
            return count;
        }

        public int subarraysDivByK(int[] nums, int k) {
            int sum = 0;
            int count = 0;
            // map from sum to count of such prefix sum
            final Map<Integer, Integer> candidates = new HashMap<>();
            candidates.put(0, 1);

            for (int i = 0; i < nums.length; i++) {
                sum = (sum + nums[i]) % k;
                if (sum < 0)
                    sum += k;

                count += candidates.getOrDefault(sum, 0);
                candidates.put(sum, candidates.getOrDefault(sum, 0) + 1);
            }
            return count;
        }

        // nums are non-negative, k can be 0 and negative
        // sum needs to be multiple of k
        public boolean checkSubarraySum(int[] nums, int k) {
            int sum = 0;
            // map from sum to oldest index of such prefix sum
            final Map<Integer, Integer> candidates = new HashMap<>();
            candidates.put(0, -1);

            for (int i = 0; i < nums.length; i++) {
                sum += nums[i];
                if (k != 0)
                    sum %= k;

                // check if at least 2 contagious elements makde that sum
                if (candidates.containsKey(sum)) {
                    if ((i - candidates.get(sum)) >= 2)
                        return true;
                } else
                    candidates.put(sum, i);
            }
            return false;
        }

        public int numSubarrayProductLessThanK(int[] nums, int k) {
            if (nums == null | nums.length == 0 || k == 0) {
                return 0;
            }
            // sliding a shrinking/growing window
            int count = 0;
            int i = 0, j = 0;
            long prod = 1;

            // forward window slide to right
            for (j = 0; j < nums.length; j++) {
                prod *= nums[j];
                // if product is already more than equals to k then shrink the window on left
                while (i <= j && prod >= k) {
                    prod /= nums[i++];
                }

                // at this point we have a contagious subarray between i and j
                // that has product less than k
                // as all numbers are positive, if the product of entire contagious
                // subarray is les than k, then each of the prefix contagious subarray
                // of this array also has product less than k.
                // for a subarry of length (j-i+1) has (j-i+1) such subarrays
                count += (j - i + 1);
            }

            return count;
        }

        public int maxProductSubArr(int a[]) {
            int localMax = 1;
            int localMin = 1;
            int globalMaxProd = 1;

            for (int i = 0; i < a.length; i++) {
                if (a[i] == 0) {
                    localMax = 1;
                    localMin = 1;
                } else if (a[i] > 0) {
                    localMax *= a[i];
                    localMin = Math.min(localMin * a[i], 1);
                } else {
                    int temp = localMin;
                    localMin = Math.min(localMax * a[i], 1);
                    localMax = Math.max(temp * a[i], 1);
                }

                globalMaxProd = Math.max(globalMaxProd, localMax);
            }

            return globalMaxProd;
        }
    }

    class Median {

        public double getStreamMedian(final int[] stream) {
            double median = 0;
            final PriorityQueue<Integer> left = new PriorityQueue<Integer>(16, Collections.reverseOrder());
            final PriorityQueue<Integer> right = new PriorityQueue<Integer>(16);

            for (int i = 0; i < stream.length; i++) {
                median = getMedian(stream[i], median, left, right);
            }
            return median;
        }

        // insert current element to the left or right heap and get median so far
        public double getMedian(final int current, final double med, final PriorityQueue<Integer> left,
                final PriorityQueue<Integer> right) {
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

        public double findMedianSortedArrays(int[] nums1, int[] nums2) {
            return findMedianSortedArrays1(nums1, nums2);
        }

        public double findMedianSortedArrays1(int A[], int B[]) {
            int m = A.length;
            int n = B.length;

            if ((m + n) % 2 != 0) // odd
                return (double) findKth(A, 0, m - 1, B, 0, n - 1, (m + n) / 2);
            else { // even
                return (findKth(A, 0, m - 1, B, 0, n - 1, (m + n) / 2)
                        + findKth(A, 0, m - 1, B, 0, n - 1, (m + n) / 2 - 1)) * 0.5;
            }
        }

        public int findKth(int A[], int p1, int r1, int B[], int p2, int r2, int k) {
            int n1 = r1 - p1 + 1;
            int n2 = r2 - p2 + 1;

            // base cases
            if (n1 == 0) {
                return B[p2 + k];
            }
            if (n2 == 0) {
                return A[p1 + k];
            }
            //
            if (k == 0) {
                return Math.min(A[p1], B[p2]);
            }

            // select two index i,j from A and B respectively such that If A[i] is between
            // B[j] and B[j-1]
            // Then A[i] would be the i+j+1 smallest element because.
            // Therefore, if we choose i and j such that i+j = k-1, we are able to find the
            // k-th smallest element.
            int i = n1 / (n1 + n2) * k;// let's try tp chose a middle element close to kth element in A
            int j = k - 1 - i;

            // add the offset
            int mid1 = Math.min(p1 + i, r1);
            int mid2 = Math.min(p2 + j, r2);

            // mid1 is greater than mid2. So, median is either in A[p1...mid1] or in
            // B[mid2+1...r2].
            // we have already see B[p2..mid2] elements smaller than kth smallest
            if (A[mid1] > B[mid2]) {
                k = k - (mid2 - p2 + 1);
                r1 = mid1;
                p2 = mid2 + 1;
            }
            // mid2 is greater than or equal mid1. So, median is either in A[mid1+1...r1] or
            // in B[p2...mid2].
            // we have already see A[p1..mid1] elements smaller than kth smallest
            else {
                k = k - (mid1 - p1 + 1);
                p1 = mid1 + 1;
                r2 = mid2;
            }

            return findKth(A, p1, r1, B, p2, r2, k);
        }

        public int findKth2(int[] A, int p1, int r1, int[] B, int p2, int r2, int k) {

            // base cases
            // if A is exhausted so return B's kth smallest
            if (p1 > r1) {
                return B[p2 + k];
            }
            // or if B is exhausted so return A's kth smallest
            if (p2 > r2) {
                return A[p1 + k];
            }

            // middle points
            int q1 = (p1 + r1) / 2;
            int q2 = (p2 + r2) / 2;
            // left partition sizes
            int m1 = q1 - p1 + 1;
            int m2 = q2 - p2 + 1;

            // combination left partition doesn'mergedRank include kth smallest
            if (m1 + m2 < k) {
                // left partition of B is smaller than kth smallest, so discard it.
                // we are discarding m2 smaller elements, so search for (k-m2)th smallest
                if (A[m1] > B[m2]) {
                    return findKth2(A, p1, r1, B, q2 + 1, r2, k - m2);
                }
                // left partition of A is smaller than the kth smallest, so discard it
                // we are discarding m1 smaller elements, so search for (k-m1)th smallest
                else {
                    return findKth2(A, q1 + 1, r1, B, p2, r2, k - m1);
                }
            } else {
                // right partition of A is larger than kth smallest, so discard it.
                if (A[m1] > B[m2]) {
                    return findKth2(A, p1, q1 - 1, B, q2 + 1, r2, k);
                }
                // right partition of B is larger than the kth smallest, so discard it
                else {
                    return findKth2(A, q1, r1, B, p2, q2 - 1, k);
                }
            }
        }

        public int medianInSortedMatrix(int[][] A) {
            int n = A.length;
            int m = A[0].length;

            if ((n * m) % 2 == 0) {
                int mid1 = kthSmallestElement(A, n / 2 - 1);
                int mid2 = kthSmallestElement(A, n / 2 + 1);
                return (mid1 + mid2) / 2;
            } else {
                return kthSmallestElement(A, n / 2);
            }
        }

        public int kthSmallestElement(int[][] A, int k) {
            int n = A.length;
            int m = A[0].length;
            MatrixElement kthSmallest = null;

            PriorityQueue<MatrixElement> minHeap = new PriorityQueue<MatrixElement>();

            // add column 0 into meanHeap - O(nlgn)
            for (int i = 0; i < n; i++) {
                minHeap.offer(new MatrixElement(A[i][0], i, 0));
            }

            // extract min from minheap and insert next element from the same row of the
            // extracted min
            int count = 0;
            while (!minHeap.isEmpty() && count < k) {
                kthSmallest = minHeap.poll();
                count++;
                //
                if (kthSmallest.col + 1 < m) {
                    minHeap.offer(new MatrixElement(A[kthSmallest.row][kthSmallest.col + 1], kthSmallest.row,
                            kthSmallest.col + 1));
                }
            }

            return kthSmallest.val;
        }
        
        
        /**
         * You are given two integer arrays nums1 and nums2 sorted in ascending order and an integer k.
         * Define a pair (u,v) which consists of one element from the first array and one element from the second array.
         * Find the k pairs (u1,v1),(u2,v2) ...(uk,vk) with the smallest sums.

            Example 1:
            
            Input: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
            Output: [[1,2],[1,4],[1,6]] 
            Explanation: The first 3 pairs are returned from the sequence: 
             [1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]
         * @param nums1
         * @param nums2
         * @param k
         * @return
         */
        public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
            // this is a merge sort problem of mering k list where each list is the row of the 
            // cross product of two arrays
            List<List<Integer>> res = new ArrayList<>();
            if(nums1.length == 0 || nums2.length == 0){
                return res;
            }
            
            // a minHeap
            PriorityQueue<int[]> pq = new PriorityQueue<>( (p1, p2) -> nums1[p1[0]]+nums2[p1[1]] - nums1[p2[0]] - nums2[p2[1]] );
            
            // as arrays are in sorted order to pairing first k from nums1 to each of nums2 is sufficient
            // first push first pair (with nums2[0]) for each of the number from nums1
            for(int i = 0; i < Math.min(k, nums1.length); i ++){
                pq.offer(new int[]{i, 0});
            }
            
            int count = 0;
            // now bfs in pq - fetching first k elements is sufficient as arrays are already sorted
            while(!pq.isEmpty() && count < k){
                int[] index = pq.poll();
                List<Integer> r = new ArrayList<>();
                r.add(nums1[index[0]]);r.add(nums2[index[1]]);
                res.add(r);
                count++;
                
                // now push next smallest pair by incrementing nums2 index
                // if it reaches the end of nums2 then we don't offer anything
                if(index[1] < nums2.length - 1){
                    pq.offer(new int[]{index[0], 1+index[1]});
                }
            }
            
            return res;
        }
    }
    
    class RemoveElements {
        public int removeElementsInPlace(int[] nums, int val) {
            int count = 0;
            for (int i = 0; i < nums.length; i++) {
                if (nums[i] == val) {
                    count++;
                } else {
                    nums[i - count] = nums[i];
                }
            }

            return nums.length - count;
        }

        public int removeDuplicatesFromSortedArray(int[] nums) {
            int count = 0;
            for (int j = 1; j < nums.length; j++) {
                // find duplicate spaces
                if (nums[j - 1] == nums[j])
                    count++;
                // start filling up spaces with next values
                // note: meantime the code reaches next value
                // we have already incremented j with total 'count' duplicated prev element
                // so, start shifting the new values by count on left
                else
                    nums[j - count] = nums[j];
            }

            // total spaces is equal to count.
            // so the final length is short of this count
            return (nums.length - count);
        }

        public ListNode removeElementsFromList(ListNode head, int val) {
            ListNode dummyHead = new ListNode(-1);
            dummyHead.next = head;
            ListNode cur = dummyHead;

            while (cur.next != null) {
                if (cur.next.val == val) {
                    cur.next = cur.next.next;
                } else {
                    cur = cur.next;
                }
            }

            return dummyHead.next;
        }
    }

    class LongestCommonPrefix {
        public String longestCommonPrefix(String[] strs) {
            if (strs == null | strs.length == 0) {
                return "";
            }
            // prefix tree or check each suffix manually
            String prefix = strs[0];
            for (int i = 1; i < strs.length; i++) {
                // prefix must start at 0
                // if no such prefix then check the prefic of prefix
                while (strs[i].indexOf(prefix) != 0) {
                    prefix = prefix.substring(0, prefix.length() - 1);
                }
            }

            return prefix;
        }
    }

    class Sorting {

        public void sortColorsDNF(int[] a) {
            int p1 = 0;
            int p2 = 0;
            int p3 = a.length - 1;

            while (p2 <= p3) {
                if (a[p2] == 0) {
                    swap(a, p2, p1);
                    p1++;
                    p2++;
                } else if (a[p2] == 1) {
                    p2++;
                } else if (a[p2] == 2) {
                    swap(a, p2, p3);
                    p3--;
                }
            }
        }

        public void merge(int[] nums1, int m, int[] nums2, int n) {
            if (nums1.length < (m + n) || nums2.length < n) {
                return;
            }

            // 3 pointer solution - make space on the left of nums 1
            // if the value is higher than nums2 in those positions
            // we can make space by swappingg it with the end
            int i = m - 1, j = n - 1, k = m + n - 1;
            while (i >= 0 && j >= 0) {
                // move the larger number at the end of nums1
                if (nums1[i] > nums2[j]) {
                    nums1[k--] = nums1[i--];
                } else {
                    nums1[k--] = nums2[j--];
                }
            }

            while (j >= 0) {
                nums1[k--] = nums2[j--];
            }
        }

        /**
         * Given an array nums, write a function to move all 0's to the end of 
         * it while maintaining the relative order of the non-zero elements.

                Example:
                
                Input: [0,1,0,3,12]
                Output: [1,3,12,0,0]
         * @param nums
         */
        public void moveZeroes(int[] nums) {
            int left = -1, i = 0, n = nums.length;

            while (i < n) {
                if (nums[i] != 0) {
                    if (left != -1) {
                        swap(nums, i, left);
                        left++;
                    }
                } else if (left == -1) {
                    left = i;
                }

                i++;
            }
        }

        public void topologicalSort(int u, ArrayList<ArrayList<Integer>> adjList, int[] visited, Stack<Integer> stack) {
            // mark as visited
            visited[u] = 1;

            // first visit all the neighbors to ensure topo sort order
            for (int v : adjList.get(u)) {
                if (visited[v] == 0) {
                    topologicalSort(v, adjList, visited, stack);
                }
            }

            stack.add(u);
        }

        public void topologicalSort(ArrayList<ArrayList<Integer>> adjList) {
            int[] visited = new int[adjList.size()];
            Stack<Integer> stack = new Stack<Integer>();

            for (int i = 0; i < adjList.size(); i++) {
                if (visited[i] == 0) {
                    topologicalSort(i, adjList, visited, stack);
                }
            }

            System.out.print("topo sort: ");
            while (!stack.isEmpty()) {
                System.out.print(stack.pop() + " ");
            }
            System.out.println();
        }
        
        /**
         * 
         * Given a string, sort it in decreasing order based on the frequency of characters.

            Example 1:
            
            Input:
            "tree"
            
            Output:
            "eert"

         * 
         * @param s
         * @return
         */
        public String frequencySort(String s) {
            Map<Character, Integer> freqMap = new HashMap<>();
            // compute freq
            for (char c : s.toCharArray()) {
                freqMap.put(c, 1+freqMap.getOrDefault(c, 0));
            }

            // now bucket sort the frequencies
            // each bucket index is the freq and values are all the charcters with the freq
            // min frequencey is 1 and max frerquency is the length of the string
            List<Character>[] buckets = new List[s.length()+1];
            for(char c : freqMap.keySet()){
                int freq = freqMap.get(c);
                if(buckets[freq] == null)
                    buckets[freq] = new ArrayList<>();
                buckets[freq].add(c); 
            }
            
            // now construct the string by repeating the charcter of each bucket freq times
            char[] res = new char[s.length()];
            int j = 0;
            for(int i = buckets.length-1; i >= 0; i--){
                if(buckets[i] != null) {
                    for(char c: buckets[i]){
                        Arrays.fill(res, j, j + i, c);
                        j += i;
                    }
                }
            }
            
            return new String(res);
        }

        public int minDiffElements(int a1[], int a2[]) {
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
            while (i < n1 && j < n2) {
                diff = Math.abs(a1[i] - a2[j]);
                if (diff < minDiff) {
                    minDiff = diff;
                    min1 = a1[i];
                    min2 = a2[j];
                }

                if (a1[i] < a2[j]) {
                    i++;
                } else {
                    j++;
                }
            }

            System.out.println(
                    "min diff between two array elements: between " + min1 + " and " + min2 + " min diff: " + minDiff);
            return minDiff;
        }
        
        /**
         * Given a list of non-negative integers nums, arrange them such that they form the largest number.
         * 
         * Input: nums = [10,2]
         * Output: "210"
         * 
         * Input: nums = [3,30,34,5,9]
         * Output: "9534330"
         * 
         * @param nums
         * @return
         */
        public String largestNumber(int[] nums) {
            String[] numsStr = new String[nums.length];
            boolean allZero = true;
            for(int i = 0; i<nums.length; i++){
                if(nums[i] > 0){
                    allZero = false;
                }
                numsStr[i] = nums[i]+"";
            }
            
            if(allZero){
                return "0";
            }
            
            Arrays.sort(numsStr, new Comparator<String>() {

                @Override
                public int compare(String o1, String o2) {
                    return (o2+o1).compareTo(o1+o2);
                }
            });
            
            StringBuilder sb = new StringBuilder();
            for(int i = 0; i<numsStr.length; i++){
                sb.append(numsStr[i]);
            }
            
            return sb.toString();
        }

        public void wiggleSort(int a[]) {
            for (int i = 0; i < a.length; i++) {
                int odd = i & 1;
                if (odd == 1) {
                    if (a[i - 1] > a[i]) {
                        swap(a, i - 1, i);
                    }
                } else {
                    if (i != 0 && a[i - 1] < a[i]) {
                        swap(a, i - 1, i);
                    }
                }
            }
        }

        /**
         *     Given an unsorted array nums, reorder it in-place such that nums[0] <= nums[1] >= nums[2] <= nums[3]....
         *     For example, given nums = [3, 5, 2, 1, 6, 4], one possible answer is [1, 6, 2, 5, 3, 4], another could be [3, 5, 1, 6, 2, 4].
         * @param str
         * @return
         */
        public String rearrangeAdjacentDuplicates(String str) {
            /*
             * Basically, A[0] <= A[1] >= A[2] <= A[3] >= A[4] <= A[5] 
             * Let's look into the problem closely. We can see if two consecutive elements are 
             * in wiggle sort order i.e. A[i-1]<=A[i]>=A[i+1] then it’s neighbors are also 
             * in wiggle order. So we could actually check by even and odd positions –
             * 
                    A[even] <= A[odd],
                    A[odd] >= A[even].

             */
            final class CharFreq implements Comparable<CharFreq> {
                char c;
                int freq;

                public CharFreq(char ch, int count) {
                    c = ch;
                    freq = count;
                }

                @Override
                public int compareTo(CharFreq o) {
                    int comp = Double.compare(freq, o.freq);
                    if (comp == 0) {
                        comp = Character.compare(o.c, c);
                    }

                    return comp;
                }
            }

            int n = str.length();
            StringBuffer rearranged = new StringBuffer();
            PriorityQueue<CharFreq> maxHeap = new PriorityQueue<CharFreq>(256, Collections.reverseOrder());
            int freqHistoGram[] = new int[256];
            // build the character frequency histogram
            for (char c : str.toCharArray()) {
                freqHistoGram[c]++;

                // if a character repeats more than n/2 then we can't rearrange
                if (freqHistoGram[c] > (n + 1) / 2) {
                    return str;
                }
            }
            // build the max heap of histogram
            for (char i = 0; i < 256; i++) {
                if (freqHistoGram[i] > 0)
                    maxHeap.add(new CharFreq(i, freqHistoGram[i]));
            }

            // rearrange - pop top 2 most frequent items and arrange them in adjacent
            // positions
            // decrease the histogram frequency of the selected chars
            while (!maxHeap.isEmpty()) {
                // extract top one and decrease the hstogram by one
                CharFreq first = maxHeap.poll();
                rearranged.append(first.c);
                first.freq--;

                CharFreq second = null;
                // extract second top and decrease the histogram by one
                if (!maxHeap.isEmpty()) {
                    second = maxHeap.poll();
                    rearranged.append(second.c);
                    second.freq--;
                }

                // add back the updated histograms
                if (first.freq > 0) {
                    maxHeap.add(first);
                }
                if (second != null && second.freq > 0) {
                    maxHeap.add(second);
                }
            }

            return rearranged.toString();
        }

        public int countTriangleTriplets(int[] segments) {
            int count = 0;
            int n = segments.length;
            Arrays.sort(segments);

            for (int i = 0; i < n - 2; i++) {
                int k = i + 2;
                for (int j = i + 1; j < n; j++) {
                    while (k < n && segments[i] + segments[j] > segments[k]) {
                        k++;
                    }
                    count += k - j - 1;
                }
            }
            return count;
        }
    }

    class MergeSortApps {

        public void rotateRight(int[] A, int i, int j) {
            int temp = A[j];
            System.arraycopy(A, i, A, i + 1, j - i);
            A[i] = temp;
        }

        public void mergeInPlace(int[] A, int i, int j) {
            while (i < j && i < A.length && j < A.length) {
                if (A[i] > A[j]) {
                    rotateRight(A, i, j);
                    i++;
                    j++;
                } else {
                    i++;
                }
            }
        }

        public void mergeSortInPlace(int[] A, int i, int j) {
            if (i >= j) {
                return;
            }
            int k = (i + j) / 2;

            mergeSortInPlace(A, i, k);
            mergeSortInPlace(A, k + 1, j);
            mergeInPlace(A, i, k + 1);
        }

        public void merge(int[] A, int i, int j, int k) {
            int[] B = new int[A.length];
            System.arraycopy(A, 0, B, 0, A.length);

            for (int r = 0; r <= k && i < A.length && j < A.length; r++) {
                if (B[i] > B[j]) {
                    A[r] = B[j++];
                } else {
                    A[r] = B[i++];
                }
            }
            System.out.println("");
        }

        public void mergeSort(int[] A, int i, int j) {
            if (i >= j) {
                return;
            }
            int k = (i + j) / 2;

            mergeSort(A, i, k);
            mergeSort(A, k + 1, j);
            merge(A, i, k + 1, A.length - 1);
        }

        public void mergeToCountSmallerOnRight(int A[], int rank[], int p, int q, int r, int count[]) {
            int n = r - p + 1;
            int i = p;
            int j = q + 1;
            int mid = q;
            int k = 0;
            int mergedRank[] = new int[n];
            int smallerCount = 0;
            while (i <= mid && j <= r) {
                // satisfies i<j, A[i]<A[j] -- so count smaller on right
                if (A[rank[i]] < A[rank[j]]) {
                    count[rank[i]] += smallerCount;
                    mergedRank[k++] = rank[i++];
                }
                // i<j, A[i]>=A[j]
                else {
                    smallerCount++;
                    mergedRank[k++] = rank[j++];
                }
            }

            // copy leftovers from the two partitions into merge
            while (i <= mid) {
                count[rank[i]] += r - q;
                mergedRank[k++] = rank[i++];
            }
            while (j <= r) {
                mergedRank[k++] = rank[j++];
            }

            // update rank
            System.arraycopy(mergedRank, 0, rank, p, n);
        }

        public void countSmallerOnRightWithMerge(int A[], int rank[], int p, int r, int count[]) {
            if (A.length == 1) {
                return;
            }

            if (p < r) {
                int q = (p + r) / 2;
                // sort left side and count ic
                countSmallerOnRightWithMerge(A, rank, p, q, count);
                // sort right side and count ic
                countSmallerOnRightWithMerge(A, rank, q + 1, r, count);
                // merge left and right and count cross ic
                mergeToCountSmallerOnRight(A, rank, p, q, r, count);
            }
        }

        public int[] countSmallerOnRightWithMerge(int A[]) {
            int n = A.length;
            int[] rank = new int[n];
            int count[] = new int[n];

            for (int i = 0; i < n; i++) {
                rank[i] = i;
            }

            countSmallerOnRightWithMerge(A, rank, 0, n - 1, count);

            return count;
        }

        // merge two sorted array A[0..q] and A[q+1..r] and return inversion count of
        // each position
        public int mergeWithInvCount(int A[], int p, int q, int r) {
            int crossInversionCount = 0;

            int n = r - p + 1;
            int i = p;
            int j = q + 1;
            int mid = q;
            int k = 0;
            int merged[] = new int[n];
            while (i <= mid && j <= r) {
                // satisfies i<j, A[i]<=A[j] -- so no inversion
                if (A[i] <= A[j]) {
                    merged[k++] = A[i++];
                } else {
                    // i<j, A[i]>A[j] --- inversion count for A[j]
                    crossInversionCount += (mid - i + 1);
                    merged[k++] = A[j++];
                }
            }

            // copy leftovers from the two partitions into merge
            while (i <= mid) {
                merged[k++] = A[i++];
            }
            while (j <= r) {
                merged[k++] = A[j++];
            }

            // update A
            System.arraycopy(merged, 0, A, p, n);

            return crossInversionCount;
        }

        public int mergeSortWithInvCount(int A[], int p, int r) {
            int inversionCount = 0;

            if (A.length == 1) {
                return 0;
            }

            if (p < r) {
                int q = (p + r) / 2;
                // sort left side and count ic
                inversionCount = mergeSortWithInvCount(A, p, q);
                // sort right side and count ic
                inversionCount += mergeSortWithInvCount(A, q + 1, r);

                // merge left and right and count cross ic
                inversionCount += mergeWithInvCount(A, p, q, r);
            }

            return inversionCount;
        }

        public class MaxSurpasser {
            int[] A, rank, surp, mergedRank;

            private MaxSurpasser(int[] a) {
                this.A = a;
                this.rank = new int[a.length];
                this.surp = new int[a.length];
                this.mergedRank = new int[a.length];
                for (int i = 0; i < rank.length; i++) {
                    rank[i] = i;
                }
            }

            public int find(int[] a) {
                return new MaxSurpasser(a).sort();
            }

            private int sort() {
                mergeSort(0, A.length - 1);
                int max = 0;
                System.out.print("bigger on rights count: ");
                for (int i = 0; i < A.length; i++) {
                    System.out.print(surp[i] + ", ");
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
                // divide
                int q = (l + r) / 2;
                mergeSort(l, q);
                mergeSort(q + 1, r);
                // conquer
                int i = l;
                int j = q + 1;
                int acc = 0;
                // accumulate through merge
                for (int s = l; s <= r; s++) {
                    if (j <= r && (i > q || A[rank[i]] < A[rank[j]])) {
                        mergedRank[s] = rank[j];
                        acc++;
                        j++;
                    } else {
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
    }

    class LargestRectangleArea {
        // O(n)
        public int largestRectangleArea(int[] hist) {
            if (hist == null || hist.length == 0) {
                return 0;
            }

            int maxArea = 0;
            int n = hist.length;
            // for a given bar the max area is incorporated by the area
            // between the lfirst higher bar on the left and first higher bar
            // on the right

            int[] left = new int[n];
            int[] right = new int[n];
            left[0] = -1;
            right[n - 1] = n;

            // find first higher bar on the left
            for (int i = 1; i < n; i++) {
                int j = i - 1;
                while (j >= 0 && hist[j] >= hist[i]) {
                    // instead of scanning all left we can skip any bar that is
                    // lower than current. That is we can use left array itself
                    // to jump throught index
                    // j--;
                    j = left[j];
                }

                left[i] = j;
            }

            // find first higher bar on the right
            for (int i = n - 2; i >= 0; i--) {
                int j = i + 1;
                while (j < n && hist[j] >= hist[i]) {
                    // instead of scanning all left we can skip any bar that is
                    // lower than current. That is we can use left array itself
                    // to jump throught index
                    // j++;
                    j = right[j];
                }

                right[i] = j;
            }

            // now compute max area from all area with each bar as the pivot
            for (int i = 0; i < n; i++) {
                // There are right[i]-left[i]-1 bars between left[i] and right[i]
                maxArea = Math.max(maxArea, hist[i] * (right[i] - left[i] - 1));
            }

            return maxArea;
        }

        public int maximalRectangle(int[][] matrix) {
            /**
             * We can transform this problem into a set of Maximum Area Rectangle in a
             * histogram sub-problems, one for each row. We will convert each of the row
             * into a histogram such that the height of the bars is equal to the consecutive
             * no of 1’s above it. For example,
             * 
             * A = 0 1 1 0 1 | _ 1 0 2 1 0 |_ | |_ 2 0 3 2 1 | | | | |_ |_|_|_|_|_|___ 2 0 3
             * 2 1
             */

            if (matrix.length == 0) {
                return 0;
            }

            int maxArea = 0;
            final int n = matrix.length;
            final int m = matrix[0].length;
            final int histograms[][] = new int[n][m];
            // convert each of the row into a histogram such that the height of
            // the bars is equal to the consecutive no of 1's above.
            // first intialize the top row with the input array
            for (int j = 0; j < m; j++) {
                histograms[0][j] = (matrix[0][j] == 1 ? 1 : 0);
            }

            // now compute the histograms.
            for (int i = 1; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    histograms[i][j] = ((matrix[i][j] == 1) ? histograms[i - 1][j] + matrix[i][j] : 0);
                }
            }

            // now we have total n histograms, one for each row.
            // Calculate the max area rectangle for each of this histogram.
            for (int i = 0; i < n; i++) {
                maxArea = Math.max(maxArea, largestRectangleArea(histograms[i]));
            }

            return maxArea;
        }
    }

    class RegexMatch {
        // correct
        public boolean isRegexMatch(String s, String p) {
            /**
             * 1, If p.charAt(j) == s.charAt(i) : dp[i][j] = dp[i-1][j-1]; 2, If p.charAt(j)
             * == '.' : dp[i][j] = dp[i-1][j-1]; 3, If p.charAt(j) == '*': here are two sub
             * conditions: 1 if p.charAt(j-1) != s.charAt(i) : dp[i][j] = dp[i][j-2] //in
             * this case, a* only counts as empty 2 if p.charAt(i-1) == s.charAt(i) or
             * p.charAt(i-1) == '.': dp[i][j] = dp[i-1][j] //in this case, a* counts as
             * multiple a or dp[i][j] = dp[i][j-1] // in this case, a* counts as single a or
             * dp[i][j] = dp[i][j-2] // in this case, a* counts as empty
             * 
             */
            if (s == null || p == null) {
                return false;
            }
            boolean[][] dp = new boolean[s.length() + 1][p.length() + 1];
            dp[0][0] = true;
            for (int i = 0; i < p.length(); i++) {
                if (p.charAt(i) == '*' && dp[0][i - 1]) {
                    dp[0][i + 1] = true;
                }
            }
            for (int i = 0; i < s.length(); i++) {
                for (int j = 0; j < p.length(); j++) {
                    if (p.charAt(j) == '.') {
                        dp[i + 1][j + 1] = dp[i][j];
                    }
                    if (p.charAt(j) == s.charAt(i)) {
                        dp[i + 1][j + 1] = dp[i][j];
                    }
                    if (p.charAt(j) == '*') {
                        if (p.charAt(j - 1) != s.charAt(i) && p.charAt(j - 1) != '.') {
                            dp[i + 1][j + 1] = dp[i + 1][j - 1];
                        } else {
                            dp[i + 1][j + 1] = (dp[i + 1][j] || dp[i][j + 1] || dp[i + 1][j - 1]);
                        }
                    }
                }
            }
            return dp[s.length()][p.length()];
        }

        public boolean isMatchWildcard(String str, String pattern) {
            int s = 0, p = 0, match = 0, starIdx = -1;
            while (s < str.length()) {
                // advancing both pointers
                if (p < pattern.length() && (pattern.charAt(p) == '?' || str.charAt(s) == pattern.charAt(p))) {
                    s++;
                    p++;
                }
                // * found, only advancing pattern pointer
                else if (p < pattern.length() && pattern.charAt(p) == '*') {
                    starIdx = p;
                    match = s;
                    p++;
                }
                // last pattern pointer was *, advancing string pointer
                else if (starIdx != -1) {
                    p = starIdx + 1;
                    match++;
                    s = match;
                }
                // current pattern pointer is not star, last patter pointer was not *
                // characters do not match
                else
                    return false;
            }

            // check for remaining characters in pattern
            while (p < pattern.length() && pattern.charAt(p) == '*')
                p++;

            return p == pattern.length();
        }

        public boolean isRegexMatch2(String str, String pat) {
            // base cases
            if (str == null) {
                return pat == null;
            } else if (pat == null) {
                return str == null;
            }
            if (str.isEmpty()) {
                return pat.isEmpty();
            }

            // pattern without *
            if ((pat.length() == 1 && pat.charAt(0) != '*') || pat.length() > 1 && pat.charAt(1) != '*') {
                // must match the first character
                if (!matchesFirst(str, pat)) {
                    return false;
                }
                // match rest
                String restStr = str.length() > 1 ? str.substring(1) : null;
                String restPat = pat.length() > 1 ? pat.substring(1) : null;
                return isRegexMatch(restStr, restPat);
            }
            // pattern with * (0 or more matches)
            else {
                // zero match of first character of the pattern
                String rigtpat = pat.length() > 2 ? pat.substring(2) : null;
                if (isRegexMatch(str, rigtpat)) {
                    return true;
                }
                // Otherwise match all possible length prefix of str to match and return true if
                // any match found
                while (matchesFirst(str, pat)) {
                    str = str.length() > 1 ? str.substring(1) : null;
                    if (isRegexMatch(str, rigtpat)) {
                        return true;
                    }
                }
            }

            return false;
        }

        private boolean matchesFirst(String str, String pat) {
            if (str == null) {
                return pat == null;
            }
            if (pat == null) {
                return str == null;
            }
            return (str.length() > 0 && str.charAt(0) == pat.charAt(0)) || (pat.charAt(0) == '.' && !str.isEmpty());
        }
    }

    class TortoiseAndHaer {

        public boolean detectCycle(ListNode head) {
            ListNode slow = head;
            ListNode fast = head.next;

            while (slow != null && fast != null && slow != fast) {
                if (fast.next == null) {
                    break;
                }
                slow = slow.next;
                fast = fast.next.next;
            }

            if (slow != null && fast != null && slow == fast) {
                return true;
            }

            return false;
        }

        public void removeCycle(ListNode head) {
            ListNode slow = head;
            ListNode fast = head.next;

            while (fast != null && fast.next != null) {
                if (slow == fast) {
                    break;
                }
                slow = slow.next;
                fast = fast.next.next;
            }

            if (slow == fast) {
                slow = head;
                while (slow != fast.next) {
                    slow = slow.next;
                    fast = fast.next;
                }

                fast.next = null;
            }
        }

        // find the single number that duplicates one or more times in an array in O(1)
        // space and O(n) time without modifying the array
        public int findDuplicate(int[] nums) {
            // using Tortoise & Hair algorithm by Donald Knuth to find cycle in A sequence.
            // This algorithm also called Floyd's cycle detection algorithm
            int n = nums.length;
            int tortoise = n;
            int hair = n;

            do {
                tortoise = nums[tortoise - 1];
                hair = nums[nums[hair - 1] - 1];
            } while (hair != tortoise);

            // find the starting point of the cycle and distance from the front, mu
            int mu = 0;
            tortoise = n;
            while (hair != tortoise) {
                tortoise = nums[tortoise - 1];
                hair = nums[hair - 1];
                mu++;
            }

            // find the min length lambda of the cycle
            int lambda = 1;
            hair = nums[tortoise - 1];

            while (hair != tortoise) {
                hair = nums[hair - 1];
                lambda++;
            }

            System.out.println("mu : " + mu + " lambda: " + lambda);

            return tortoise;
        }
    }

    class MinMax {

        public int[] slidingWindowMax(final int[] in, final int w) {
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

        public int[] slidingWindowMin(final int[] in, final int w) {
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
    }

    class StreamOps {

        public char firstUnique(char[] stream) {
            HashSet<Character> seen = new HashSet<Character>();
            LinkedHashSet<Character> uniques = new LinkedHashSet<Character>();

            for (int i = 0; i < stream.length; i++) {
                char c = Character.toLowerCase(stream[i]);
                if (!seen.contains(c)) {
                    seen.add(c);
                    uniques.add(c);
                } else {
                    uniques.remove(c);
                }
            }

            if (uniques.size() > 0) {
                return uniques.iterator().next();
            } else
                return '\0';
        }

        boolean equals(JSONTokenStream s1, JSONTokenStream s2) {

            JsonNode node1 = null;
            JsonNode node2 = null;
            Stack<JsonNode> lastRoot = new Stack<>();
            while (s1.hasNext()) {
                JSONToken cur = s1.next();
                if (node1 == null && cur.type() == 0) {
                    node1 = new JsonNode(cur);
                    lastRoot.push(node1);
                } else if (node1 != null && cur.type() == 1) {
                    lastRoot.pop();
                } else {
                    lastRoot.peek().addChild(cur);
                }
            }

            lastRoot = new Stack<>();
            while (s2.hasNext()) {
                JSONToken cur = s2.next();
                if (node2 == null && cur.type() == 0) {
                    node2 = new JsonNode(cur);
                    lastRoot.push(node2);
                } else if (node2 != null && cur.type() == 1) {
                    lastRoot.pop();
                } else {
                    lastRoot.peek().addChild(cur);
                }
            }

            return node1.equals(node2);
        }

        public class MovingAvgLastN {
            int maxTotal;
            int total;
            double lastN[];
            double avg;
            int head;

            public MovingAvgLastN(int N) {
                maxTotal = N;
                lastN = new double[N];
                avg = 0;
                head = 0;
                total = 0;
            }

            public void add(double num) {
                double prevSum = total * avg;

                if (total == maxTotal) {
                    prevSum -= lastN[head];
                    total--;
                }

                head = (head + 1) % maxTotal;
                int emptyPos = (maxTotal + head - 1) % maxTotal;
                lastN[emptyPos] = num;

                double newSum = prevSum + num;
                total++;
                avg = newSum / total;
            }

            public double getAvg() {
                return avg;
            }
        }

    }

    class StockOptimization {
        /**
         * Say you have an array for which the ith element is the price of a given stock on day i.
         * If you were only permitted to complete at most one transaction 
         * (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.
         * Note that you cannot sell a stock before you buy one.
         * 
         * Input: [7,1,5,3,6,4]
            Output: 5
            Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
            Not 7-1 = 6, as selling price needs to be larger than buying price.


         * @param price
         * @return
         */
        public int maxProfit(int[] price) {
            if(price.length == 0){
                return 0;
            }
            int maxProfit = 0;
            int minBuy = 0;

            for (int i = 0; i < price.length; i++) {
                if (price[i] < price[minBuy]) {
                    minBuy = i;
                }
                if ((price[i] - price[minBuy]) > maxProfit) {
                    maxProfit = price[i] - price[minBuy];
                }
            }

            return maxProfit;
        }
        
        public int maxProfitTradeMany(int[] price) {
            if(price.length == 0){
                return 0;
            }
            int maxProfit = 0;
            int minBuy = 0;
            int maxSell = 0;

            int i = 0;
            while (i < price.length-1) {
                // in order to maximize profit we will be greedy
                // we don't buy if the stock pricing is falling 
                // but we opportunisically buy just before stock price starts to rise again
                // imagine if we had an oracle telling us what will be the price of the stock next day
                while(i < price.length - 1 && price[i+1] <= price[i]){
                    i++;
                }
                // set the local minima for buy
                minBuy = i;
                
                // in order to maximize profit we will be greedy
                // we don't sell immediately if the stock price is raising 
                // but we opportunisically sell just before stock price starts to fall again
                // imagine if we had an oracle telling us what will be the price of the stock next day
                while(i < price.length - 1 && price[i+1] > price[i]){
                    i++;
                }
                // set the local minima for buy
                maxSell = i;
                
                // collect all profits
                maxProfit += price[maxSell] - price[minBuy];
            }

            return maxProfit;
        }
        
        class MinStack {

            Stack<Integer> stack;
            int min = Integer.MAX_VALUE;
            /** initialize your data structure here. */
            public MinStack() {
                stack = new Stack<>();
            }
            
            public void push(int x) {
                // if this is a new min then we are changing the minimum
                // we record the prev minumim by an additional push.
                if(x <= min){
                    stack.push(min);
                    min = x;
                }
                // then push x 
                // make sure we pop in the reverse order
                stack.push(x);
            }
            
            public void pop() {
                int x = stack.pop();
                // if we arre popping min then take the second minumum
                // second minium wss pushed when minium value was chnaged
                if (x == min){
                    min = stack.pop();
                }
            }
            
            public int top() {
                return stack.peek();
            }
            
            public int getMin() {
                return min;
            }
        }
    }

    class SocialMedia {
        public int influencer(int[][] following) {
            int influencer = 0;

            // find the candidate influencer by testing each person i
            // a person i may be a candidate influencer if s/he follows nobody or som
            for (int i = 1; i < following.length; i++) {
                if (following[i][influencer] == 0 || following[influencer][i] == 1) {
                    influencer = i;
                }
            }

            // verify that the candidate influencer is indeed an influencer
            for (int i = 0; i < following.length; i++) {
                if (i == influencer) {
                    continue;
                }
                // to be influencer he/she shouldn't follow anybody and there should be nobody
                // else who doesn't follw him/her
                if (following[i][influencer] == 0 || following[influencer][i] == 1) {
                    return -1;
                }
            }

            return influencer;
        }
    }
    class KthSmallest {
        public int kthLargest(int[] nums, int k) {
            // kth largest is (n-k+1) th smallest
            return kthSmallest(nums, 0, nums.length-1, nums.length - k + 1);
        }
        
        public int kthSmallest(int nums[], int l, int h, int k){
            if(l > h){
                return -1;
            }
            // partition the array with respect to a random pivot.
            // if there are exactly k elements less than equal to the pivot (left partition) 
            // then pivot is the kth smallest. If more elements on left then keep searching recursively
            // for kth on left partition. If less elements on the left then find look for the remaining (k-leftSize) to the right
            
            int pivotIndex = partition(nums, l, h);
            // total number of elements on the left partition
            int n = pivotIndex - l + 1;
            
            if(k == n){
                // this is thr kth
                return nums[pivotIndex];
            }
            // more elements on left - search on left
            else if(n > k){
                return kthSmallest(nums, l, pivotIndex - 1, k);
            }
            // less elements on left , so find  (k-n) th on the right
            else{
                return kthSmallest(nums, pivotIndex + 1, h, k-n);
            }
        }
        
        public int partition(int nums[], int l, int h){
            int p = l-1;
            int i = l;
            
            // take a random element as pivot - ideally the pivot should be a median
            int pivotIndex = h;//(int) Math.round(l + Math.random() * (h - l + 1));
            
            // swap the pivot at the end
            //swap(nums, pivotIndex, h);
            
            // now loop from low to high and parrtition the array into two
            // such that x <= pivot is on the left of the partition p and 
            // x > pivot stays on the right
            for(i = l; i < h; i++) {
                if(nums[i] <= nums[pivotIndex]){
                    swap(nums, ++p, i);
                }
            }
            
            // swap the pivot to it's own position 
            swap(nums, p+1, h);
            // return the pivot index
            return p+1;
        }
        
        private int medianOfMedianPartition(int nums[], int l, int h){
            int pivot = medianOfMedians(nums, l, h);
            swap(nums, h, pivot);
            return partition(nums, l, h);
        }
        
        private int medianOfMedians(int A[], int left, int right){
            final int numMedians = Math.round((right - left) / 5);
            for (int i = 0; i < numMedians; i++) {
                // get the median of the five-element subgroup
                final int subLeft = left + i * 5;
                int subRight = subLeft + 4;
                if (subRight > right) {
                    subRight = right;
                }
                // alternatively, use a faster method that works on lists of size 5
                final int q = (subRight - subLeft) / 2;
                swap(A, q, subRight);
                final int medianIdx = partition(A, subLeft, subRight);
                // move the median to a contiguous block at the beginning of the list
                swap(A, left + i, medianIdx);
            }
            // select the median from the contiguous block
            final int q = numMedians / 2;
            swap(A, q, right);
            
            return partition(A, left, left + numMedians - 1);
        }
    }
    
    public static void main(String[] args) {

        Test t = new Test();
        Sorting st = t.new Sorting();
        st.merge(new int[] { 1, 2, 3, 0, 0, 0 }, 3, new int[] { 2, 5, 6 }, 3);
        
        BinarySearch bsss = t.new BinarySearch();
        bsss.searchRange(new int[] { 5, 7, 7, 8, 8, 10 }, 8);
        
        Median m = t.new Median();
        m.kSmallestPairs(new int[] {1, 2}, new int[] {3}, 3);
        
        
        ListNode myhead = t.new ListNode(1);
        ListNode mydum = t.new ListNode(0);
        mydum.next = myhead;
        myhead.next = t.new ListNode(2);
        myhead = myhead.next;
        myhead.next = t.new ListNode(3);
        myhead = myhead.next;
        myhead.next = t.new ListNode(4);
        myhead = myhead.next;
        myhead.next = t.new ListNode(5);
        
        LinkedListOps ll = t.new LinkedListOps();
        ll.rotateListRight(mydum.next, 1);
        
        Sorting ss = t.new Sorting();
        ss.frequencySort("his s he a ha he  ha ae");
        
        //["Trie","insert","insert","insert","insert","insert","insert","search","search","search","search","search","search","search","search","search","startsWith","startsWith","startsWith","startsWith","startsWith","startsWith","startsWith","startsWith","startsWith"]
        //[[],["app"],["apple"],["beer"],["add"],["jam"],["rental"],["apps"],["app"],["ad"],["applepie"],["rest"],["jan"],["rent"],["beer"],["jam"],["apps"],["app"],["ad"],["applepie"],["rest"],["jan"],["rent"],["beer"],["jam"]]

        Trie tr = t.new GraphTraversals().new Trie();
        tr.insert("app");
        tr.insert("apple");
        tr.insert("apps");
        tr.insert("app");
        tr.search("app");
        tr.search("apple");
        tr.startsWith("app");
        tr.startsWith("ad");
        
        LRUCache lru = t.new LRUCache(2);
        lru.put(1, 1);
        lru.put(2, 2);
        lru.get(1);
        lru.put(3, 3);
        lru.get(2);
        lru.put(3, 3);
        lru.get(1);
        lru.get(2);
        lru.get(4);
        
        LinkedListOps liops = t.new LinkedListOps();
        liops.reverseBetween(mydum.next, 2, 4);
        
        DP dp = t.new DP();
        dp.canJump(new int[] {5,9,3,2,1,0,2,3,3,1,0,0});
        
        WalkBFS w = t.new WalkBFS();
        //w.wordLadderAll(new HashSet<>(Arrays.asList(new String[] {"ted","tex","red","tax","tad","den","rex","pee"})), "red", "tax");
        //w.ladderLength2("ymain", "oecij", Arrays.asList(new String[] {"ymann","yycrj","oecij","ymcnj","yzcrj","yycij","xecij","yecij","ymanj","yzcnj","ymain"}));
        w.wordLadderAll(new HashSet<>(Arrays.asList(new String[] {"ymann","yycrj","oecij","ymcnj","yzcrj","yycij","xecij","yecij","ymanj","yzcnj","ymain"})), "ymain", "oecij");

        // String minl = t.minLenSuperSubString1("ADOBECODEBANC", "ABC");

        DPGrid dpg = t.new DPGrid();
        int msp = dpg.minPathSum(new int[][] { { 1, 3, 1 }, { 1, 5, 1 }, { 4, 2, 1 } });

        Subsequences ssq = t.new Subsequences();
        int lis = ssq.lengthOfLIS(new int[] { -2, -1 });

        NumericalComputation nc = t.new NumericalComputation();
        int ft = nc.fourListSum(new int[] { -1, -1 }, new int[] { -1, 1 }, new int[] { -1, 1 }, new int[] { 1, -1 });

        ThreeSum tsum = t.new ThreeSum();
        List<List<Integer>> rest = tsum.kSum(new int[] { 1, 0, -1, 0, -2, 2 }, 0, 4);

        int re = tsum.minDiffElement(new int[] { 1, 3, 6, 7, 9, 10 }, -1);

        BinarySearchTree bst = t.new BinarySearchTree();
        BinarySearch bs = t.new BinarySearch();
        int re1 = bs.binarySearchClosest(new int[] { 1, 3, 6, 7, 9, 10 }, 0, 5, -1);

        IntervalOps iops = t.new IntervalOps();
        PartitionLabels s = iops.new PartitionLabels();
        @SuppressWarnings("unused")
        List<Integer> parts = s.partitionLabels("ababcbacadefegdehijhklij");

        int floor = bs.floor(new int[] { -1, 0, 3, 3, 5, 6, 8 }, 4);
        int ceil = bs.ceil(new int[] { -1, 0, 3, 3, 5, 6, 8 }, 4);

        ListNode dummy = null;
        ListNode head = t.new ListNode(4);
        dummy = head;
        head.next = t.new ListNode(2);
        head = head.next;
        head.next = t.new ListNode(1);
        head = head.next;
        head.next = t.new ListNode(3);
//        head = head.next;
//        head.next = t.new ListNode(5);
//        head = head.next;
//        head.next = t.new ListNode(7);
//        head = head.next;
//        head.next = t.new ListNode(6);
//        head = head.next;
//        head.next = t.new ListNode(8);

        LinkedListOps llo = t.new LinkedListOps();
        ListNode sorted = llo.MergeSortList(dummy);

        // ListNode head2 = t.new DLLListToBSTInplace(dummy).convert();

        head = llo.reverseK(dummy, null, null, 3, 0);

        /**
         * 4 / \ 2 6 / \ / \ 1 3 5 7 \ 8
         * 
         * 
         */

        head = llo.oddEvenList(dummy);
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

        // List<Integer> llist = t.preorderTraversal(root6);

        // TreeNode dll = t.inorderDLListInplace(root6);
        // TreeNode dll = t.inorderDLListInplaceRecursive(root6);
        BinaryTree bt = t.new BinaryTree();
        TreeNode dll = bt.inorderCircularDLListInplace(root6);

        TreeNode tail = dll;
        while (dll != null && dll != tail) {
            System.out.print(dll.val + " ");
            tail = dll;
            dll = dll.right;
        }

        System.out.println();
        while (tail != null) {
            System.out.print(tail.val + " ");
            tail = tail.left;
        }

        nc.validIp(Arrays.asList(new String[] {
                "ewdjbwouhfsu255.248.89.9sdssdadsa0.0.0.0.sdbkjdb1.34.46.7wdfdsfsd23.34.56.sfdfsdfs00.0.0.0.0.dfsfs" }));

        Interval[] ints = new Interval[] { t.new Interval(1, 11, 5), t.new Interval(2, 6, 7), t.new Interval(3, 13, 9),
                t.new Interval(12, 7, 16), t.new Interval(14, 3, 25), t.new Interval(19, 18, 22),
                t.new Interval(23, 13, 29), t.new Interval(24, 4, 28) };

        Interval[] ov = iops.mergeOverlappedIntervals(ints);

        List<List<Integer>> resultt = new ArrayList<>();
        System.out.println(resultt);
        List<Integer> curr = new ArrayList<>();

        int[] res = bs.searchRange(new int[] { 5, 7, 7, 8, 8, 10 }, 8);
        System.out.println(res);

        int m1[][] = new int[][] { { 9, 9, 4 }, { 6, 6, 8 }, { 2, 1, 0 } };
        BackTrack btr = t.new BackTrack();
        List<Integer> ers = btr.walkDFS(m1);

        Graph g = t.new Graph();
        g.vertices = new int[] { 2, 3, 4, 6, 5, 7, 4, 1, 8, 3 };
        g.edges = new Edge[g.vertices.length + 1][g.vertices.length + 1];

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

        GraphTraversals gt = t.new GraphTraversals();
        List<Integer> spath = gt.shortestPath(0, 9);

        List<List<String>> input = Arrays.asList(Arrays.asList(new String[] { "a", "b", "c" }),
                Arrays.asList(new String[] { "d", "e" }), Arrays.asList(new String[] { "f" }));
        List<List<String>> result = new ArrayList<>();
        String[] cur = new String[input.size()];

        PermutationCombinatons pc = t.new PermutationCombinatons();
        pc.permList(input, cur, 0, result);
    }
}
