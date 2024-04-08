import Jama.*;

ArrayList<PVector> points;
ArrayList<PVector> ellipseCenters;
ArrayList<PVector> ellipseAxes;
ArrayList<Float> ellipseAngles;
PVector mean;
PVector[] eigenvectors;
float[] eigenvalues;

void setup() {
  size(800, 800);
  points = new ArrayList<PVector>();
  ellipseCenters = new ArrayList<PVector>();
  ellipseAxes = new ArrayList<PVector>();
  ellipseAngles = new ArrayList<Float>();
  eigenvectors = new PVector[2];
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
  float rx = random(10, 50);
  float ry = random(10, 50);
  for (float t = 0; t < TWO_PI; t += PI/2) {
    float x = center.x + rx * cos(t) * cos(angle) - ry * sin(t) * sin(angle);
    float y = center.y + rx * cos(t) * sin(angle) + ry * sin(t) * cos(angle);
    points.add(new PVector(x, y));
  }
  ellipseCenters.add(center);
  ellipseAxes.add(new PVector(rx, ry));
  ellipseAngles.add(angle);
}

void calculatePCA() {
  // Calculate mean
  mean = new PVector(0, 0);
  for (PVector point : points) {
    mean.add(point);
  }
  mean.div(points.size());

  // Build covariance matrix
  Matrix cov = new Matrix(2, 2);
  for (PVector point : points) {
    PVector diff = PVector.sub(point, mean);
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        cov.set(i, j, cov.get(i, j) + diff.array()[i] * diff.array()[j]);
      }
    }
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
    ellipse(point.x, point.y, 10, 10);
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
  line(mean.x, mean.y, mean.x + eigenvectors[0].x * sqrt(eigenvalues[0]) * 1, mean.y + eigenvectors[0].y * sqrt(eigenvalues[0]) * 1);
  stroke(0, 0, 255);
  line(mean.x, mean.y, mean.x + eigenvectors[1].x * sqrt(eigenvalues[1]) * 1, mean.y + eigenvectors[1].y * sqrt(eigenvalues[1]) * 1);
}

void drawEllipse() {
  pushMatrix();
  translate(mean.x, mean.y);
  rotate(atan2(eigenvectors[0].y, eigenvectors[0].x));
  noFill();
  stroke(0, 255, 0);
  ellipse(0, 0, sqrt(eigenvalues[0]) * 2, sqrt(eigenvalues[1]) * 2);
  popMatrix();
}
