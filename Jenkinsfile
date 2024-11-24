pipeline {
    agent any
    stages {
        stage('Checkout') {
            steps {
                git url: 'https://github.com/your-repo.git'
            }
        }
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t enhanced-attribute-classifier:${BUILD_NUMBER} .'
            }
        }
        stage('Push to Docker Registry') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'docker-credentials', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    sh '''
                    echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin
                    docker tag enhanced-attribute-classifier:${BUILD_NUMBER} your-repo/enhanced-attribute-classifier:${BUILD_NUMBER}
                    docker push your-repo/enhanced-attribute-classifier:${BUILD_NUMBER}
                    '''
                }
            }
        }
        stage('Deploy to Kubernetes') {
            steps {
                sh 'kubectl apply -f kubernetes/deployment.yaml'
            }
        }
    }
}

