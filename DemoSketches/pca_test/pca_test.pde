import Jama.*;

ArrayList<PVector> points;
ArrayList<PVector> ellipseCenters;
ArrayList<PVector> ellipseAxes;
ArrayList<Float> ellipseAngles;
ArrayList<Matrix> ellipsecovs;
PVector mean;
PVector[] eigenvectors;
float[] eigenvalues;

Matrix inriaCov;

float deviations =1.41;

void setup() {
  size(800, 800);
  points = new ArrayList<PVector>();
  ellipseCenters = new ArrayList<PVector>();
  ellipseAxes = new ArrayList<PVector>();
  ellipseAngles = new ArrayList<Float>();
  eigenvectors = new PVector[2];
  ellipsecovs = new ArrayList<Matrix>();
  inriaCov = new Matrix(2, 2);
}

void draw() {
  background(255);
  if (points.size() > 0) {
    calculatePCA();
    drawPoints();
    drawEllipses();
    drawPCA();
    drawEllipse();
  }
}

void mousePressed() {
  PVector center = new PVector(mouseX, mouseY);
  float angle = random(TWO_PI);
  float rx = random(50, 100);
  float ry = random(50, 100);
  points.add(new PVector(center.x, center.y));
  for (float t = 0; t < TWO_PI; t += PI/2) {
    float x = center.x + rx * cos(t) * cos(angle) - ry * sin(t) * sin(angle);
    float y = center.y + rx * cos(t) * sin(angle) + ry * sin(t) * cos(angle);
    points.add(new PVector(x, y));
  }
  Matrix S = new Matrix(2, 2);
  Matrix R = new Matrix(2, 2);
  S.set(0, 0, rx); S.set(0, 1, 0.0); S.set(1, 0, 0.0); S.set(1, 1, ry);
  R.set(0, 0, cos(angle)); R.set(0, 1, -sin(angle)); R.set(1, 0, sin(angle)); R.set(1, 1, cos(angle));
  
  Matrix cov = new Matrix(2, 2);
  cov = R.transpose().times(S.transpose()).times(S).times(R);
  
  ellipseCenters.add(center);
  ellipseAxes.add(new PVector(rx, ry));
  ellipseAngles.add(angle);
  ellipsecovs.add(cov);
}

void calculatePCA() {
  // Calculate mean
  mean = new PVector(0, 0);
  for (PVector point : points) {
    mean.add(point);
  }
  mean.div(points.size());
  inriaCov.set(0, 0, 0); inriaCov.set(1, 0, 0); inriaCov.set(0, 1, 0); inriaCov.set(1, 1, 0);

  // Build covariance matrix
  Matrix cov = new Matrix(2, 2);
  for (PVector point : points) {
    PVector diff = PVector.sub(point, mean);
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        cov.set(i, j, cov.get(i, j) + diff.array()[i] * diff.array()[j]);
      }
    }
    
    Matrix md = new Matrix(2, 2);
    md.set(0, 0, diff.x * diff.x); md.set(1, 0, diff.x * diff.y); md.set(0, 1, diff.x * diff.y); md.set(1, 1, diff.y * diff.y);
    
  }
  cov.timesEquals(1.0 / points.size());

  // Calculate eigenvectors and eigenvalues
  EigenvalueDecomposition eig = cov.eig();
  eigenvalues = new float[2];
  eigenvalues[0] = (float)eig.getRealEigenvalues()[0];
  eigenvalues[1] = (float)eig.getRealEigenvalues()[1];
  eigenvectors[0] = new PVector((float)eig.getV().get(0, 0), (float)eig.getV().get(1, 0));
  eigenvectors[1] = new PVector((float)eig.getV().get(0, 1), (float)eig.getV().get(1, 1));
  
  
}

void drawPoints() {
  fill(0);
  for (PVector point : points) {
    strokeWeight(1);
    stroke(0, 0, 0);
    ellipse(point.x, point.y, 3, 3);
  }
}

void drawEllipses() {
  for (int i = 0; i < ellipseCenters.size(); i++) {
    PVector center = ellipseCenters.get(i);
    PVector axes = ellipseAxes.get(i);
    float angle = ellipseAngles.get(i);
    pushMatrix();
    translate(center.x, center.y);
    rotate(angle);
    noFill();
    stroke(128, 128, 128);
    ellipse(0, 0, axes.x * 2, axes.y * 2);
    popMatrix();
  }
}

void drawPCA() {
  stroke(255, 0, 0);
  line(mean.x, mean.y, mean.x + eigenvectors[0].x * sqrt(eigenvalues[0]) * deviations, mean.y + eigenvectors[0].y * sqrt(eigenvalues[0]) * deviations);
  stroke(0, 0, 255);
  line(mean.x, mean.y, mean.x + eigenvectors[1].x * sqrt(eigenvalues[1]) * deviations, mean.y + eigenvectors[1].y * sqrt(eigenvalues[1]) * deviations);
}

void drawEllipse() {
  pushMatrix();
  translate(mean.x, mean.y);
  rotate(atan2(eigenvectors[0].y, eigenvectors[0].x));
  noFill();
  strokeWeight(5);
  stroke(0, 255, 0);
  ellipse(0, 0, sqrt(eigenvalues[0]) * 2 * deviations, sqrt(eigenvalues[1]) * 2 * deviations);
  strokeWeight(4);
  popMatrix();
}
