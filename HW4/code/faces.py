

# python libraries
import collections

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# libraries specific to project
import util
from util import *
from cluster import *

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y) :
    """
    Translate images to (labeled) points.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets

    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """

    n,d = X.shape

    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in xrange(n) :
        images[y[i]].append(X[i,:])

    points = []
    for face in images :
        count = 0
        for im in images[face] :
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def plot_clusters(clusters, title, average) :
    """
    Plot clusters along with average points of each cluster.

    Parameters
    --------------------
        clusters -- ClusterSet, clusters to plot
        title    -- string, plot title
        average  -- method of ClusterSet
                    determines how to calculate average of points in cluster
                    allowable: ClusterSet.centroids, ClusterSet.medoids
    """

    plt.figure()
    np.random.seed(20)
    label = 0
    colors = {}
    centroids = average(clusters)
    for c in centroids :
        coord = c.attrs
        plt.plot(coord[0],coord[1], 'ok', markersize=12)
    for cluster in clusters.members :
        label += 1
        colors[label] = np.random.rand(3,)
        for point in cluster.points :
            coord = point.attrs
            plt.plot(coord[0], coord[1], 'o', color=colors[label])
    plt.title(title)
    plt.show()


def generate_points_2d(N, seed=1234) :
    """
    Generate toy dataset of 3 clusters each with N points.

    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed

    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)

    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]

    label = 0
    points = []
    for m,s in zip(mu, sigma) :
        label += 1
        for i in xrange(N) :
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))

    return points


######################################################################
# k-means and k-medoids
######################################################################

def random_init(points, k) :
    """
    Randomly select k unique elements from points to be initial cluster centers.

    Parameters
    --------------------
        points         -- list of Points, dataset
        k              -- int, number of clusters

    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2c: implement (hint: use np.random.choice)
    initial_points = np.random.choice(a=points,size=k, replace=False)
    return initial_points
    ### ========== TODO : END ========== ###


def cheat_init(points) :
    """
    Initialize clusters by cheating!

    Details
    - Let k be number of unique labels in dataset.
    - Group points into k clusters based on label (i.e. class) information.
    - Return medoid of each cluster as initial centers.

    Parameters
    --------------------
        points         -- list of Points, dataset

    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2f: implement
    #find k
    label_dict = {}
    for p in points:
        if p.label in label_dict:
            label_dict[p.label].append(p)
        else:
            label_dict[p.label] = [p]
    initial_points = []
    for k,v in label_dict.iteritems():
        cluster = Cluster(v)
        initial_points.append(cluster.medoid())
    return initial_points
    ### ========== TODO : END ========== ###

def find_assignment(points,centroids): #centroids have a labels
    """return a dictionary, key: centroid(cluster), value: list of points assigned to that cluster"""
    assignment_dict = {}
    for p in points:
        min_dist, minIndex = np.inf, -1
        for index in range(len(centroids)):
            if p.distance (centroids[index]) < min_dist:
                min_dist = p.distance (centroids[index])
                minIndex = index
        #now we have the min_index and min_dist
        if minIndex in assignment_dict:
            assignment_dict[minIndex].append(p)
        else:
            assignment_dict[minIndex] = [p]
    return assignment_dict




def kMeans(points, k, init='random', plot=False) :
    """
    Cluster points into k clusters using variations of k-means algorithm.

    Parameters
    --------------------
        points  -- list of Points, dataset
        k       -- int, number of clusters
        average -- method of ClusterSet
                   determines how to calculate average of points in cluster
                   allowable: ClusterSet.centroids, ClusterSet.medoids
        init    -- string, method of initialization
                   allowable:
                       'cheat'  -- use cheat_init to initialize clusters
                       'random' -- use random_init to initialize clusters
        plot    -- bool, True to plot clusters with corresponding averages
                         for each iteration of algorithm

    Returns
    --------------------
        k_clusters -- ClusterSet, k clusters
    """

    ### ========== TODO : START ========== ###
    # part 2c: implement
    # Hints:
    #   (1) On each iteration, keep track of the new cluster assignments
    #       in a separate data structure. Then use these assignments to create
    #       a new ClusterSet object and update the centroids.
    #   (2) Repeat until the clustering no longer changes.
    #   (3) To plot, use plot_clusters(...).

    #first get the centroids
    current_centroids = []
    prev_clusters = None
    if init == "random":
        current_centroids = random_init(points, k)
    else:
        current_centroids = cheat_init(points)
    #create k clusters
    iter = 1
    while True: #loop until the current cluster is equivalent to previous cluster
        cur_cluster_set = ClusterSet()
        for k,v in find_assignment(points,current_centroids).iteritems():
            cur_cluster_set.add(Cluster(v))
        """
        cur_cluster_set = ClusterSet([Cluster(v) for _, v
                                       in find_assignment_kMeans(
                                           points, current_centroids).iteritems()])
        """
        if plot == True:
            str = "K_means Cluster Assignments And Cluster Centers For Iteration %i, Initialization Label = %s" % (iter,init)
            plot_clusters(cur_cluster_set,str,ClusterSet.centroids)
        if prev_clusters is not None and cur_cluster_set.equivalent(prev_clusters) == True:
            return cur_cluster_set
        else:
            iter += 1
            prev_clusters = cur_cluster_set
            current_centroids = cur_cluster_set.centroids()


    k_clusters = ClusterSet()
    return k_clusters
    ### ========== TODO : END ========== ###


def kMedoids(points, k, init='random', plot=False) :
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    ### ========== TODO : START ========== ###
    # part 2e: implement
    #steps
    #find 3 medoids, same as before
    #enter the for loop (while True)
    #get clusters
    #for each cluster, find the medoid
    #repeat until the last clusterSet and current are equivalent and you have converged
    current_medoids = random_init(points,k) if init == "random" else cheat_init(points)
    prev_cluster_set = None
    iter = 1
    while True:
        cur_cluster_set = ClusterSet()
        for k,v in find_assignment(points,current_medoids).iteritems():
            cur_cluster_set.add(Cluster(v))
        if plot == True:
            str = "K_medoid Cluster Assignments And Cluster Centers For Iteration %i, Initialization Label = %s" % (iter,init)
            plot_clusters(cur_cluster_set,str,ClusterSet.medoids)
        if prev_cluster_set is not None and cur_cluster_set.equivalent(prev_cluster_set) == True: #if we are done
            return cur_cluster_set
        else:
            current_medoids = cur_cluster_set.medoids()
            prev_cluster_set = cur_cluster_set
            iter += 1


    k_clusters = ClusterSet()
    return k_clusters
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################

def main() :
    ### ========== TODO : START ========== ###
    # part 1: explore LFW data set
    X, y = get_lfw_data()
    #show_image(np.mean(X, axis=0)) #axis 0 is finds the average of the column for all of the images
    U,mu = util.PCA(X)
    """
    l_values = [1,10, 50,100,500, 1288]
    image_arr = np.arange(start=0, stop=12)
    for l in l_values:
        Z, Ul = util.apply_PCA_from_Eig(X, U, l, mu)  # to lower the dimension of the images
        X_rec = reconstruct_from_PCA(Z,Ul,mu)
        title = "Reconstructed images for l = %d" % (l)
        print title
        plot_gallery(X_rec, title= title)
    """

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part 2d-2f: cluster toy dataset
    #part 2d
    np.random.seed(1234)
    points = generate_points_2d(N=20)
    print("2d")
    #uncomment this
    """
    clusters = kMeans(points,3,'random',True)
    #end of 2d
    #part 2e
    medoid_cluster = kMedoids(points,3,'random',True)
    #end of 2e
    #part 2f, cheat initialization
    kmeans_cheat = kMeans(points,3,'cheat',True)
    
    kmedoid_cheat = kMedoids(points,3,'cheat', True)
    """
    ### ========== TODO : END ========== ###



    #IMPORTANT
    #Begin of 3a comment
    """
    ### ========== TODO : START ========== ###
    # part 3a: cluster faces
    np.random.seed(1234)
    X1, y1 = util.limit_pics(X, y, [4, 6, 13, 16], 40)
    points = build_face_image_points(X1, y1)
    kmean_score_list = []
    kmedoid_score_list = []

    for i in np.arange(0,10):
        kmean_cluster = kMeans(points, 4, 'random', False)
        kmean_score_list.append(kmean_cluster.score())
        kmedoid_cluster = kMedoids(points, 4, 'random', False)
        kmedoid_score_list.append(kmedoid_cluster.score())
    kmean_avg = np.mean(kmean_score_list)
    kmean_max = max(kmean_score_list)
    kmean_min = min(kmean_score_list)

    kmedoid_avg = np.mean(kmedoid_score_list)
    kmedoid_max = max(kmedoid_score_list)
    kmedoid_min = min(kmedoid_score_list)
    """
    #End of 3A comments

    #IMPORTANT
    #Begin of 3b comment
    """"
    # part 3b: explore effect of lower-dimensional representations on clustering performance
    print ("3b")
    np.random.seed(1234)
    # Use PCA to get the the eigenfaces (and eigenvectors)
    U, mu = util.PCA(X)
    l_range = np.arange(1,42)
    k = 2
    X2, y2 = util.limit_pics(X, y, [4, 13], 40)
    kmean_score_dict = {}
    kmedoid_score_dict = {}
    for l in l_range:
        Z1, Ul1 = apply_PCA_from_Eig(X2,U,l, mu) #reduce the dimension
        X2_reconstructed = reconstruct_from_PCA(Z1,Ul1,mu)
        points = build_face_image_points(X2_reconstructed, y2)
        kmeans_clust = kMeans(points,k,'cheat',False)
        kmedoid_clust = kMedoids(points,k,'cheat',False)
        kmean_score_dict[l] = kmeans_clust.score()
        kmedoid_score_dict[l] = kmedoid_clust.score()
    print "3b here"
    plt.plot(list(kmean_score_dict.keys()),list(kmean_score_dict.values()), color= 'b', label='kMeans')
    plt.plot(list(kmedoid_score_dict.keys()),list(kmedoid_score_dict.values()),color= 'g', label='kMedoid')
    plt.title("kMean and kMedoid Scores vs l (Number of Principal Components)")
    plt.xlabel("l (Number of Principal Components)")
    plt.ylabel("kMean and kMedoid Scores")
    plt.legend()
    plt.show()
    #End of 3b comment
    """

    # part 3c: determine ``most discriminative'' and ``least discriminative'' pairs of images
    np.random.seed(1234)
    min_score = (np.inf, None, None)
    max_score = (-1, None, None)
    #we know there are 19 people
    print ("Starting")
    for i in range(0,19):
        for j in range(0,19):
            if i == j: #if on the same person
                continue
            X3, y3 = util.limit_pics(X, y, [i, j], 40) #receive the images
            points= build_face_image_points(X3, y3)
            kmedoid_clust = kMedoids(points,2, 'cheat', False)
            if kmedoid_clust.score() < min_score[0]:
                min_score = (kmedoid_clust.score,i,j)
            if kmedoid_clust.score() > max_score[0]:
                max_score = (kmedoid_clust.score,i,j)
    #now we have the min and max clusters
    print ("before the plot")
    plot_representative_images(X,y,[max_score[1],max_score[2]], title ="Images with Maximum Cluster Score (Best Clustering)")
    plot_representative_images(X,y,[min_score[1], min_score[2]], title= "Images with Minumum Cluster Score (Worst Clustering)")


    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()
