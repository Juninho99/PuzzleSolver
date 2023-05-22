#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core.hpp>

#include <iostream>
#include <algorithm>
#include <numeric>
#include <sys/stat.h>
#include <filesystem>
#include <iomanip>

using namespace cv;
using namespace std;

typedef Point_<float> Point2f;
vector< tuple<bool, vector <vector<double> >, vector <vector<double> > > > SlagaliceSviPodaci;
//VideoWriter oVideoWriter;
Mat KonacnaSlagalica(3800, 2700, CV_8UC3, Scalar(0,0,0));
int velicinaKvadrata = 700;

void izdvojiSlagalice(int debljina){
    Mat hsvUlaznaSkeniranaSlika;
    Mat maskaUlaznaSkeniranaSlika;

/*!
########################################################
    Ucitavanje slike (ucitava se cijela skenirana slika.
    *put do slike je u build fajlu projekta
    *nomenklatura: "brojSkeniraneSlike_rezolucija.jpg"
########################################################
*/
    int redniBrojSlagaliceNaSlici = 0;
    int broj_slika = 5;
for(int broj_slike = 1; broj_slike < broj_slika + 1; broj_slike++)
{
    String path = "200//" + to_string(broj_slike) + "_200.jpg";
    Mat ulaznaSkeniranaSlika = imread(path);
    cvtColor(ulaznaSkeniranaSlika, hsvUlaznaSkeniranaSlika, COLOR_BGR2HSV );

/*!
    ########################################################
        Dobijanje maske, pozadina crna, slagalice bijele
    ########################################################
*/

    Mat plavaSlika(hsvUlaznaSkeniranaSlika.rows, hsvUlaznaSkeniranaSlika.cols, CV_8UC1);
    Mat zelenaSlika(hsvUlaznaSkeniranaSlika.rows, hsvUlaznaSkeniranaSlika.cols, CV_8UC1);
    Mat crvenaSlika(hsvUlaznaSkeniranaSlika.rows, hsvUlaznaSkeniranaSlika.cols, CV_8UC1);
    for(int i = 0; i < hsvUlaznaSkeniranaSlika.rows; i++) {
        for(int j = 0; j < hsvUlaznaSkeniranaSlika.cols; j++) {
            plavaSlika.at<uchar>(i, j) = hsvUlaznaSkeniranaSlika.at<Vec3b>(i, j)[0];
            zelenaSlika.at<uchar>(i, j) = hsvUlaznaSkeniranaSlika.at<Vec3b>(i, j)[1];
            crvenaSlika.at<uchar>(i, j) = hsvUlaznaSkeniranaSlika.at<Vec3b>(i, j)[2];
        }
    }

    threshold(crvenaSlika, maskaUlaznaSkeniranaSlika, 60, 255, 0);

    erode(maskaUlaznaSkeniranaSlika, maskaUlaznaSkeniranaSlika, Mat());
    dilate(maskaUlaznaSkeniranaSlika, maskaUlaznaSkeniranaSlika, Mat());

/*!
    ########################################################
        Pronalazak kontura na skeniranoj slagalici
    ########################################################
*/
    vector<vector<Point> > kontureSlagalicaNaSkeniranojSlici;
    findContours( maskaUlaznaSkeniranaSlika, kontureSlagalicaNaSkeniranojSlici, RETR_TREE, CHAIN_APPROX_SIMPLE );
    vector<Rect> kvadratiOkoSlagalica( kontureSlagalicaNaSkeniranojSlici.size() );

    Mat obojeneKonture = Mat::zeros( ulaznaSkeniranaSlika.size(), CV_8UC3 );

    size_t maleKonture = 100;
    size_t brojKonturaNaSlici = kontureSlagalicaNaSkeniranojSlici.size();

    for( size_t i = 0; i < brojKonturaNaSlici; i++ ) {
        size_t brojTacakaNaKonturi = kontureSlagalicaNaSkeniranojSlici[i].size();

/*!
    ########################################################
        Crtanje svih slagalica na slici
    ########################################################
*/
        //Ovo ne gleda konture koji su male
        if(brojTacakaNaKonturi < maleKonture)
            continue;

        kvadratiOkoSlagalica[i] = boundingRect( kontureSlagalicaNaSkeniranojSlici[i] );
        Point centarSlagalice( kvadratiOkoSlagalica[i].x + int(kvadratiOkoSlagalica[i].width / 2),
                               kvadratiOkoSlagalica[i].y + int(kvadratiOkoSlagalica[i].height / 2) );

        vector<double> udaljenostTacakaNaKonturiOdCentra;

        /// Pronalazak grafika udaljenosti tačaka konture od centra
        int br = 0;
        for(size_t tacka = 0; tacka < kontureSlagalicaNaSkeniranojSlici[i].size(); tacka++){
            double udaljenost = norm(centarSlagalice - kontureSlagalicaNaSkeniranojSlici[i][tacka]);
            udaljenostTacakaNaKonturiOdCentra.push_back(udaljenost );
            br++;
        }

/*!
    ########################################################
        Crtanje grafika udaljenosti po tačkama
    ########################################################
*/

/*
        Size s(br, 300);
        Mat drawing = Mat::zeros( s, CV_8UC3 );
        polylines(drawing, grafikUdaljenosti, false, Scalar( 0,0,255 ), 1, 8);
        //imshow("2", drawing);
        //waitKey();
*/

/*!
    ########################################################
        Izdvajanje potencijalnih ćoškova
    ########################################################
*/
        vector<pair<double, int>> izvodi;
        pair<double, int> par;

        int brojTacaka = udaljenostTacakaNaKonturiOdCentra.size();

        int indeksig[10];
        int indeksid[10];
        double parametrig[10];
        double parametrid[10];

        for(int j = 0; j < brojTacaka ; j++) {
            for (int in = 1; in < 10; in++) {
                indeksid[in] = j - in;
                if(j - in < 0)
                    indeksid[in] = brojTacaka + j - in;

                indeksig[in] = j + in;
                if(j + in > brojTacaka - 1)
                    indeksig[in] = j - brojTacaka + in;
            }

            for (int in = 1; in < 10; in++) {
                parametrig[in] = udaljenostTacakaNaKonturiOdCentra[j] - udaljenostTacakaNaKonturiOdCentra[indeksig[in]];
                parametrid[in] = udaljenostTacakaNaKonturiOdCentra[j] - udaljenostTacakaNaKonturiOdCentra[indeksid[in]];
            }
            double sumag = 0;
            double sumad = 0;

            double sumag4 = 0;
            double sumad4 = 0;
            double sumag9 = 0;
            double sumad9 = 0;
            if(parametrig[1] >= 0 && parametrid[1] >= 0) {
                for (int in = 1; in < 10; in++) {
                    if(in == 6 || in == 5){
                        sumag = sumag + 2*parametrig[in];
                        sumad = sumad + 2*parametrid[in];
                    }
                    else {
                        sumag = sumag + parametrig[in];
                        sumad = sumad + parametrid[in];
                    }

                    if(in < 5) {
                        sumag4 = sumag4 + parametrig[in];
                        sumad4 = sumad4 + parametrid[in];
                    }
                    if(in > 6) {
                        sumag9 = sumag9 + parametrig[in];
                        sumad9 = sumad9 + parametrid[in];
                    }
                }
                if((sumag4 > 2 && sumad4 > 2) || (sumag9+sumad9>100)) {


                    par.first = sumag + sumad;
                    par.second = j;
                    izvodi.push_back(par);
                }

            }
        }


/*!
    ########################################################
        Vise tacaka zadovoljava prethodni uslov. Npr ispupcenja udubljenja
        na slagalicama također zadovoljavaju ovaj uslov, ali su izvodi veći
        kod coskova, pa onda u sljedecem korako odabiremo 4 najbolja kandidata
    ########################################################
*/
        sort(izvodi.rbegin(), izvodi.rend());

        vector<Point> coskovi;
        for(int j = 0; j < 4; j++) {
            coskovi.push_back(kontureSlagalicaNaSkeniranojSlici[i][izvodi[0].second]);
            int broj_za_usporedbu = izvodi[0].second;

            for(size_t l = 0; l < izvodi.size(); l++){
                if(izvodi[l].second >= broj_za_usporedbu - 6 && izvodi[l].second <= broj_za_usporedbu + 6) {
                    izvodi.erase(izvodi.begin() + l);
                    l--;
                }
            }
        }

        // Ovo je hardkodirano samo na ovoj slici od svih zbog specificnosti teksta na slici

        if(redniBrojSlagaliceNaSlici == 6) {
            Point brute(780, 1483);
            coskovi[1] = brute;
        }
        if(redniBrojSlagaliceNaSlici == 16) {
            Point brute(347, 436);
            coskovi[3] = brute;
        }
        if(redniBrojSlagaliceNaSlici == 33) {
            Point brute1(1586, 1189);
            coskovi[2] = brute1;
            Point brute2(1562, 982);
            coskovi[3] = brute2;
        }


/*!
    ########################################################
        nakon sto su odredjeni coskovi korisno je razlicitim bojama obojiti sve cetiri
        strane slagalice. To se radi u narednom koraku.
    ########################################################
*/

        vector<Point> gore;
        vector<Point> lijevo;
        vector<Point> dole;
        vector<Point> desno;
        vector<Point> goreOstatak;

        vector<Point> gorePravi;

        int orjenatacija = 1;
        vector<Point> sortiraniCoskovi;
        for( size_t z = 0; z < kontureSlagalicaNaSkeniranojSlici[i].size(); z++ ) {
            for(size_t h = 0; h < coskovi.size(); h++){
                if(kontureSlagalicaNaSkeniranojSlici[i][z] == coskovi[h]) {
                    sortiraniCoskovi.push_back(coskovi[h]);
                    orjenatacija++;
                }
            }
            if(orjenatacija == 1)
                gore.push_back(kontureSlagalicaNaSkeniranojSlici[i][z]);
            if(orjenatacija == 2)
                lijevo.push_back(kontureSlagalicaNaSkeniranojSlici[i][z]);
            if(orjenatacija == 3)
                dole.push_back(kontureSlagalicaNaSkeniranojSlici[i][z]);
            if(orjenatacija == 4)
                desno.push_back(kontureSlagalicaNaSkeniranojSlici[i][z]);
            if(orjenatacija == 5)
                goreOstatak.push_back(kontureSlagalicaNaSkeniranojSlici[i][z]);
        }

        Scalar color1(150, 0 , 0);
        Scalar color2(0, 150 , 0);
        Scalar color3(0, 0 , 150);
        Scalar color4(150, 0 , 150);

        Point GL(80, 1870);
        Point DD(460, 2145);

        //ovo je za snimanje videa
        Rect okvirVideo(GL, DD);

        //debljina linije
        //int debljina = 5;

        for( size_t p = 1; p < desno.size() + 1; p++ ) {
            if(p == desno.size())
                line(obojeneKonture, desno[p-1], goreOstatak[0],color1, debljina,LINE_8);
            else
                line(obojeneKonture, desno[p-1], desno[p], color1, debljina, LINE_8);

            //Mat video = obojeneKonture(okvirVideo);
            //oVideoWriter.write(video);
        }

        for( size_t p = 1; p < goreOstatak.size() + 1; p++ ) {
            if(p == goreOstatak.size()){
                if(gore.size() > 0)
                    line(obojeneKonture, goreOstatak[p-1], gore[0],color3, debljina,LINE_8);
            }
            else
                line(obojeneKonture, goreOstatak[p-1], goreOstatak[p], color3, debljina, LINE_8);
            //Mat video = obojeneKonture(okvirVideo);
            //oVideoWriter.write(video);
        }


        if(gore.size() == 0)
            line(obojeneKonture, goreOstatak[goreOstatak.size()-1], lijevo[0],color3, debljina,LINE_8);
        for( size_t p = 1; p < gore.size() + 1; p++ ) {
            if(p == gore.size())
                line(obojeneKonture, gore[p-1], lijevo[0],color3, debljina,LINE_8);
            else
                line(obojeneKonture, gore[p-1], gore[p], color3, debljina, LINE_8);
            //Mat video = obojeneKonture(okvirVideo);
            //oVideoWriter.write(video);
        }

        for( size_t p = 1; p < lijevo.size() + 1; p++ ) {
            if(p == lijevo.size())
                line(obojeneKonture, lijevo[p-1], dole[0],color2, debljina,LINE_8);
            else
                line(obojeneKonture, lijevo[p-1], lijevo[p], color2, debljina, LINE_8);
            //Mat video = obojeneKonture(okvirVideo);
            //oVideoWriter.write(video);
        }

        for( size_t p = 1; p < dole.size() + 1; p++ ) {
            if(p == dole.size())
                line(obojeneKonture, dole[p-1], desno[0],color4, debljina,LINE_8);
            else
                line(obojeneKonture, dole[p-1], dole[p], color4, debljina, LINE_8);
            //Mat video = obojeneKonture(okvirVideo);
            //oVideoWriter.write(video);
        }


        //Za spasavanje videa.
        //oVideoWriter.release();
        // return;


        for( size_t p = 0; p < goreOstatak.size(); p++) {
            gorePravi.push_back(goreOstatak[p]);
        }

        for( size_t p = 0; p < gore.size(); p++) {
            gorePravi.push_back(gore[p]);
        }

/*!
    ########################################################
        Iscrtavanje tacaka na slagalici razlicitim bojama, korisno za Armina
    ########################################################
*/
        int boja = 60;
        for(size_t h = 0; h < sortiraniCoskovi.size(); h++){
            //int radius = 0; //Declaring the radius
            Scalar line_Color(0, boja, 0);//Color of the circle
            int thickness = -1;//thickens of the line

            ///OVO DODAJE COSKOVE NA JEDNOBOJNE KONTURE
            //circle(samoKonture, sortiraniCoskovi[h],radius, line_Color, thickness);


            ///OVO DODAJE COSKOVE NA OBOJENE KONTURE
            circle(obojeneKonture, sortiraniCoskovi[h],0, line_Color, thickness);
            //circle(ulaznaSkeniranaSlika, sortiraniCoskovi[h],0, line_Color, thickness);
            circle(maskaUlaznaSkeniranaSlika, sortiraniCoskovi[h], 0, Scalar(boja), 1);

            boja = boja + 50;
        }

        Point gl(kvadratiOkoSlagalica[i].tl().x - 10, kvadratiOkoSlagalica[i].tl().y - 10);
        Point dd(kvadratiOkoSlagalica[i].tl().x + kvadratiOkoSlagalica[i].width + 10,
                 kvadratiOkoSlagalica[i].tl().y + kvadratiOkoSlagalica[i].height + 10);

        Rect okvir(gl, dd);

        String putanjaZaObojeneKonture = "200//KontureDebljina" + to_string(debljina) + "//" + to_string(redniBrojSlagaliceNaSlici + 1) + ".png";
        Mat slagalica = obojeneKonture(okvir);
        imwrite(putanjaZaObojeneKonture, slagalica);

        String putanjaZaSlagalice = "200//Slagalice//" + to_string(redniBrojSlagaliceNaSlici + 1) + ".png";
        Mat slagalicaIzdvojena = ulaznaSkeniranaSlika(okvir);
        imwrite(putanjaZaSlagalice, slagalicaIzdvojena);

        String putanjaZaMaske = "200//Maske//" + to_string(redniBrojSlagaliceNaSlici + 1) + ".png";
        Mat slagalicaMaska = maskaUlaznaSkeniranaSlika(okvir);
        imwrite(putanjaZaMaske, slagalicaMaska);
/*!
    ########################################################
        Računanje matrice podataka
    ########################################################
*/
        redniBrojSlagaliceNaSlici ++;
        //cout << redniBrojSlagaliceNaSlici <<endl;

        }
    }
}

void dodajBorderKontureMaskeSlagalice(){
    /*!
    ########################################################
        Dodavanje crnog okvira za sve konture, maske i slagalice.
        Usput, zapisivanje slika sa svim rotacijama u zasebne foldere, i za konture i za maske i za slagalice.
    ########################################################
    */

    Mat img, img90, img180, img270;
    Mat img1, img901, img1801, img2701;
    Mat img2, img902, img1802, img2702;

    mkdir("200//Konture");
    mkdir("200//Konture0");
    mkdir("200//Konture90");
    mkdir("200//Konture180");
    mkdir("200//Konture270");

    mkdir("200//SlagaliceRotirane");
    mkdir("200//SlagaliceRotirane0");
    mkdir("200//SlagaliceRotirane90");
    mkdir("200//SlagaliceRotirane180");
    mkdir("200//SlagaliceRotirane270");

    mkdir("200//MaskeRotirane");
    mkdir("200//MaskeRotirane0");
    mkdir("200//MaskeRotirane90");
    mkdir("200//MaskeRotirane180");
    mkdir("200//MaskeRotirane270");

    for(int i = 1; i < 161; i++){
        img = imread("200\\KontureDebljina5\\" + to_string(i) + ".png");
        int offset = 20;
        copyMakeBorder(img, img, offset, offset, offset, offset, BORDER_CONSTANT);

        //rotacija 90 stepeni
        transpose(img, img90);
        flip(img90, img90, 1);

        //rotacija 180 stepeni
        flip(img, img180, -1);

        //rotacija 270 stepeni
        transpose(img, img270);
        flip(img270, img270, 0);

        imwrite("200\\Konture\\" + to_string(i) + ".png", img);
        imwrite("200\\Konture0\\" + to_string(i) + ".png", img);
        imwrite("200\\Konture90\\" + to_string(i) + ".png", img90);
        imwrite("200\\Konture180\\" + to_string(i) + ".png", img180);
        imwrite("200\\Konture270\\" + to_string(i) + ".png", img270);

        //Maske i slagalice

        img1 = imread("200//Slagalice//" + to_string(i) + ".png");
        img2 = imread("200//Maske//" + to_string(i) + ".png");

        int sirina = img1.cols;
        int visina = img1.rows;

        int dodajVisina = velicinaKvadrata - visina;
        int dodajSirina = velicinaKvadrata - sirina;

        int offsetTop = dodajVisina / 2;
        int offsetDown = dodajVisina - dodajVisina / 2;
        int offsetLeft = dodajSirina / 2;
        int offsetRight = dodajSirina - dodajSirina / 2;

        copyMakeBorder(img1, img1, offsetTop, offsetDown, offsetLeft, offsetRight, BORDER_CONSTANT);
        copyMakeBorder(img2, img2, offsetTop, offsetDown, offsetLeft, offsetRight, BORDER_CONSTANT);

        //rotacija 90 stepeni
        transpose(img1, img901);
        flip(img901, img901, 1);

        transpose(img2, img902);
        flip(img902, img902, 1);

        //rotacija 180 stepeni
        flip(img1, img1801, -1);

        flip(img2, img1802, -1);

        //rotacija 270 stepeni
        transpose(img1, img2701);
        flip(img2701, img2701, 0);

        transpose(img2, img2702);
        flip(img2702, img2702, 0);

        imwrite("200\\SlagaliceRotirane\\" + to_string(i) + ".png", img1);
        imwrite("200\\SlagaliceRotirane0\\" + to_string(i) + ".png", img1);
        imwrite("200\\SlagaliceRotirane90\\" + to_string(i) + ".png", img901);
        imwrite("200\\SlagaliceRotirane180\\" + to_string(i) + ".png", img1801);
        imwrite("200\\SlagaliceRotirane270\\" + to_string(i) + ".png", img2701);

        imwrite("200\\MaskeRotirane\\" + to_string(i) + ".png", img2);
        imwrite("200\\MaskeRotirane0\\" + to_string(i) + ".png", img2);
        imwrite("200\\MaskeRotirane90\\" + to_string(i) + ".png", img902);
        imwrite("200\\MaskeRotirane180\\" + to_string(i) + ".png", img1802);
        imwrite("200\\MaskeRotirane270\\" + to_string(i) + ".png", img2702);
    }
}

void dodajSveRotacijeKontureMaskeSlagalice(){
    /*!
    ########################################################
        Prilagođavanje rotacija, tako da sve slike zadrže istu vrijednost piksela
        po određenim ćoškovima, te po konturama kada su one različite.
    ########################################################
    */
    for(int i = 1; i < 161; i++){
        Mat img90, img180, img270;
        Mat img901, img1801, img2701;
        Mat img902, img1802, img2702;

        img90 = imread("200\\Konture90\\" + to_string(i) + ".png");
        img180 = imread("200\\Konture180\\" + to_string(i) + ".png");
        img270 = imread("200\\Konture270\\" + to_string(i) + ".png");

        Mat temp90(img90.rows, img90.cols, CV_8UC3);
        Mat temp180(img180.rows, img180.cols, CV_8UC3);
        Mat temp270(img270.rows, img270.cols, CV_8UC3);

        Mat end90(temp90.rows, temp90.cols, CV_8UC3);
        Mat end180(temp180.rows, temp180.cols, CV_8UC3);
        Mat end270(temp270.rows, temp270.cols, CV_8UC3);

        for(int i = 0; i < img90.rows; i++)
        {
            for(int j = 0; j < img90.cols; j++)
            {
                Vec3b color = img90.at<Vec3b>(i, j);
                Vec3b newColor;
                if(color.val[0] == 150 && color.val[1] == 0 && color.val[2] == 0){
                    newColor[0] = 50; newColor[1] = 50; newColor[2] = 50;
                }
                else if(color.val[0] == 0 && color.val[1] == 150 && color.val[2] == 0){
                    newColor[0] = 100; newColor[1] = 100; newColor[2] = 100;
                }
                else if(color.val[0] == 0 && color.val[1] == 0 && color.val[2] == 150){
                    newColor[0] = 150; newColor[1] = 150; newColor[2] = 150;
                }
                else if(color.val[0] == 150 && color.val[1] == 0 && color.val[2] == 150){
                    newColor[0] = 200; newColor[1] = 200; newColor[2] = 200;
                }
                else if(color.val[0] == 0 && color.val[1] == 160 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 50; newColor[2] = 0;
                }
                else if(color.val[0] == 0 && color.val[1] == 110 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 100; newColor[2] = 0;
                }
                else if(color.val[0] == 0 && color.val[1] == 60 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 150; newColor[2] = 0;
                }
                else if(color.val[0] == 0 && color.val[1] == 210 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 200; newColor[2] = 0;
                }

                temp90.at<Vec3b>(i, j) = newColor;
            }
        }

        for(int i = 0; i < temp90.rows; i++)
        {
            for(int j = 0; j < temp90.cols; j++)
            {
                Vec3b color = temp90.at<Vec3b>(i, j);
                Vec3b newColor;
                if(color.val[0] == 50 && color.val[1] == 50 && color.val[2] == 50){
                    newColor[0] = 150; newColor[1] = 0; newColor[2] = 150; //90
                }
                else if(color.val[0] == 100 && color.val[1] == 100 && color.val[2] == 100){
                    newColor[0] = 0; newColor[1] = 0; newColor[2] = 150; //90
                }
                else if(color.val[0] == 150 && color.val[1] == 150 && color.val[2] == 150){
                    newColor[0] = 150; newColor[1] = 0; newColor[2] = 0; //90
                }
                else if(color.val[0] == 200 && color.val[1] == 200 && color.val[2] == 200){
                    newColor[0] = 0; newColor[1] = 150; newColor[2] = 0; //90
                }
                else if(color.val[0] == 0 && color.val[1] == 50 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 110; newColor[2] = 0; //90
                }
                else if(color.val[0] == 0 && color.val[1] == 100 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 60; newColor[2] = 0; //90
                }
                else if(color.val[0] == 0 && color.val[1] == 150 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 210; newColor[2] = 0; //90
                }
                else if(color.val[0] == 0 && color.val[1] == 200 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 160; newColor[2] = 0; //90
                }
                end90.at<Vec3b>(i, j) = newColor;
            }
        }

        for(int i = 0; i < img180.rows; i++)
        {
            for(int j = 0; j < img180.cols; j++)
            {
                Vec3b color = img180.at<Vec3b>(i, j);
                Vec3b newColor;
                if(color.val[0] == 150 && color.val[1] == 0 && color.val[2] == 0){
                    newColor[0] = 50; newColor[1] = 50; newColor[2] = 50;
                }
                else if(color.val[0] == 0 && color.val[1] == 150 && color.val[2] == 0){
                    newColor[0] = 100; newColor[1] = 100; newColor[2] = 100;
                }
                else if(color.val[0] == 0 && color.val[1] == 0 && color.val[2] == 150){
                    newColor[0] = 150; newColor[1] = 150; newColor[2] = 150;
                }
                else if(color.val[0] == 150 && color.val[1] == 0 && color.val[2] == 150){
                    newColor[0] = 200; newColor[1] = 200; newColor[2] = 200;
                }
                else if(color.val[0] == 0 && color.val[1] == 160 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 50; newColor[2] = 0;
                }
                else if(color.val[0] == 0 && color.val[1] == 110 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 100; newColor[2] = 0;
                }
                else if(color.val[0] == 0 && color.val[1] == 60 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 150; newColor[2] = 0;
                }
                else if(color.val[0] == 0 && color.val[1] == 210 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 200; newColor[2] = 0;
                }

                temp180.at<Vec3b>(i, j) = newColor;
            }
        }

        for(int i = 0; i < temp180.rows; i++)
        {
            for(int j = 0; j < temp180.cols; j++)
            {
                Vec3b color = temp180.at<Vec3b>(i, j);
                Vec3b newColor;
                if(color.val[0] == 50 && color.val[1] == 50 && color.val[2] == 50){
                    newColor[0] = 0; newColor[1] = 150; newColor[2] = 0; //180
                }
                else if(color.val[0] == 100 && color.val[1] == 100 && color.val[2] == 100){
                    newColor[0] = 150; newColor[1] = 0; newColor[2] = 0; //180
                }
                else if(color.val[0] == 150 && color.val[1] == 150 && color.val[2] == 150){
                    newColor[0] = 150; newColor[1] = 0; newColor[2] = 150; //180
                }
                else if(color.val[0] == 200 && color.val[1] == 200 && color.val[2] == 200){
                    newColor[0] = 0; newColor[1] = 0; newColor[2] = 150; //180
                }
                else if(color.val[0] == 0 && color.val[1] == 50 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 60; newColor[2] = 0; //180
                }
                else if(color.val[0] == 0 && color.val[1] == 100 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 210; newColor[2] = 0; //180
                }
                else if(color.val[0] == 0 && color.val[1] == 150 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 160; newColor[2] = 0; //180
                }
                else if(color.val[0] == 0 && color.val[1] == 200 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 110; newColor[2] = 0; //180
                }
                end180.at<Vec3b>(i, j) = newColor;
            }
        }

        for(int i = 0; i < img270.rows; i++)
        {
            for(int j = 0; j < img270.cols; j++)
            {
                Vec3b color = img270.at<Vec3b>(i, j);
                Vec3b newColor;
                if(color.val[0] == 150 && color.val[1] == 0 && color.val[2] == 0){
                    newColor[0] = 50; newColor[1] = 50; newColor[2] = 50;
                }
                else if(color.val[0] == 0 && color.val[1] == 150 && color.val[2] == 0){
                    newColor[0] = 100; newColor[1] = 100; newColor[2] = 100;
                }
                else if(color.val[0] == 0 && color.val[1] == 0 && color.val[2] == 150){
                    newColor[0] = 150; newColor[1] = 150; newColor[2] = 150;
                }
                else if(color.val[0] == 150 && color.val[1] == 0 && color.val[2] == 150){
                    newColor[0] = 200; newColor[1] = 200; newColor[2] = 200;
                }
                else if(color.val[0] == 0 && color.val[1] == 160 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 50; newColor[2] = 0;
                }
                else if(color.val[0] == 0 && color.val[1] == 110 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 100; newColor[2] = 0;
                }
                else if(color.val[0] == 0 && color.val[1] == 60 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 150; newColor[2] = 0;
                }
                else if(color.val[0] == 0 && color.val[1] == 210 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 200; newColor[2] = 0;
                }

                temp270.at<Vec3b>(i, j) = newColor;
            }
        }

        for(int i = 0; i < temp270.rows; i++)
        {
            for(int j = 0; j < temp270.cols; j++)
            {
                Vec3b color = temp270.at<Vec3b>(i, j);
                Vec3b newColor;
                if(color.val[0] == 50 && color.val[1] == 50 && color.val[2] == 50){
                    newColor[0] = 0; newColor[1] = 0; newColor[2] = 150; //270
                }
                else if(color.val[0] == 100 && color.val[1] == 100 && color.val[2] == 100){
                    newColor[0] = 150; newColor[1] = 0; newColor[2] = 150; //270
                }
                else if(color.val[0] == 150 && color.val[1] == 150 && color.val[2] == 150){
                    newColor[0] = 0; newColor[1] = 150; newColor[2] = 0; //270
                }
                else if(color.val[0] == 200 && color.val[1] == 200 && color.val[2] == 200){
                    newColor[0] = 150; newColor[1] = 0; newColor[2] = 0; //270
                }
                else if(color.val[0] == 0 && color.val[1] == 50 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 210; newColor[2] = 0; //270
                }
                else if(color.val[0] == 0 && color.val[1] == 100 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 160; newColor[2] = 0; //270
                }
                else if(color.val[0] == 0 && color.val[1] == 150 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 110; newColor[2] = 0; //270
                }
                else if(color.val[0] == 0 && color.val[1] == 200 && color.val[2] == 0){
                    newColor[0] = 0; newColor[1] = 60; newColor[2] = 0; //270
                }
                end270.at<Vec3b>(i, j) = newColor;
            }
        }

        imwrite("200\\Konture\\" + to_string(i+160) + ".png", end90);
        imwrite("200\\Konture\\" + to_string(i+320) + ".png", end180);
        imwrite("200\\Konture\\" + to_string(i+480) + ".png", end270);

        //Maske

        img901 = imread("200\\MaskeRotirane90\\" + to_string(i) + ".png");
        img1801 = imread("200\\MaskeRotirane180\\" + to_string(i) + ".png");
        img2701 = imread("200\\MaskeRotirane270\\" + to_string(i) + ".png");

        Mat temp901(img901.rows, img901.cols, CV_8UC3);
        Mat temp1801(img1801.rows, img1801.cols, CV_8UC3);
        Mat temp2701(img2701.rows, img2701.cols, CV_8UC3);

        Mat end901(temp901.rows, temp901.cols, CV_8UC3);
        Mat end1801(temp1801.rows, temp1801.cols, CV_8UC3);
        Mat end2701(temp2701.rows, temp2701.cols, CV_8UC3);

        for(int i = 0; i < img901.rows; i++)
        {
            for(int j = 0; j < img901.cols; j++)
            {
                Vec3b color = img901.at<Vec3b>(i, j);
                Vec3b newColor;
                if(color.val[0] == 160 && color.val[1] == 160 && color.val[2] == 160){
                    newColor[0] = 0; newColor[1] = 50; newColor[2] = 0;
                }
                else if(color.val[0] == 110 && color.val[1] == 110 && color.val[2] == 110){
                    newColor[0] = 0; newColor[1] = 100; newColor[2] = 0;
                }
                else if(color.val[0] == 60 && color.val[1] == 60 && color.val[2] == 60){
                    newColor[0] = 0; newColor[1] = 150; newColor[2] = 0;
                }
                else if(color.val[0] == 210 && color.val[1] == 210 && color.val[2] == 210){
                    newColor[0] = 0; newColor[1] = 200; newColor[2] = 0;
                }
                else if(color.val[0] == 255 && color.val[1] == 255 && color.val[2] == 255){
                    newColor[0] = 0; newColor[1] = 250; newColor[2] = 0;
                }

                temp901.at<Vec3b>(i, j) = newColor;
            }
        }

        for(int i = 0; i < temp901.rows; i++)
        {
            for(int j = 0; j < temp901.cols; j++)
            {
                Vec3b color = temp901.at<Vec3b>(i, j);
                Vec3b newColor;
                if(color.val[0] == 0 && color.val[1] == 50 && color.val[2] == 0){
                    newColor[0] = 110; newColor[1] = 110; newColor[2] = 110; //90
                }
                else if(color.val[0] == 0 && color.val[1] == 100 && color.val[2] == 0){
                    newColor[0] = 60; newColor[1] = 60; newColor[2] = 60; //90
                }
                else if(color.val[0] == 0 && color.val[1] == 150 && color.val[2] == 0){
                    newColor[0] = 210; newColor[1] = 210; newColor[2] = 210; //90
                }
                else if(color.val[0] == 0 && color.val[1] == 200 && color.val[2] == 0){
                    newColor[0] = 160; newColor[1] = 160; newColor[2] = 160; //90
                }
                else if(color.val[0] == 0 && color.val[1] == 250 && color.val[2] == 0){
                    newColor[0] = 255; newColor[1] = 255; newColor[2] = 255; //90
                }
                end901.at<Vec3b>(i, j) = newColor;
            }
        }

        for(int i = 0; i < img1801.rows; i++)
        {
            for(int j = 0; j < img1801.cols; j++)
            {
                Vec3b color = img1801.at<Vec3b>(i, j);
                Vec3b newColor;
                if(color.val[0] == 160 && color.val[1] == 160 && color.val[2] == 160){
                    newColor[0] = 0; newColor[1] = 50; newColor[2] = 0;
                }
                else if(color.val[0] == 110 && color.val[1] == 110 && color.val[2] == 110){
                    newColor[0] = 0; newColor[1] = 100; newColor[2] = 0;
                }
                else if(color.val[0] == 60 && color.val[1] == 60 && color.val[2] == 60){
                    newColor[0] = 0; newColor[1] = 150; newColor[2] = 0;
                }
                else if(color.val[0] == 210 && color.val[1] == 210 && color.val[2] == 210){
                    newColor[0] = 0; newColor[1] = 200; newColor[2] = 0;
                }
                else if(color.val[0] == 255 && color.val[1] == 255 && color.val[2] == 255){
                    newColor[0] = 0; newColor[1] = 250; newColor[2] = 0;
                }

                temp1801.at<Vec3b>(i, j) = newColor;
            }
        }

        for(int i = 0; i < temp1801.rows; i++)
        {
            for(int j = 0; j < temp1801.cols; j++)
            {
                Vec3b color = temp1801.at<Vec3b>(i, j);
                Vec3b newColor;
                if(color.val[0] == 0 && color.val[1] == 50 && color.val[2] == 0){
                    newColor[0] = 60; newColor[1] = 60; newColor[2] = 60; //180
                }
                else if(color.val[0] == 0 && color.val[1] == 100 && color.val[2] == 0){
                    newColor[0] = 210; newColor[1] = 210; newColor[2] = 210; //180
                }
                else if(color.val[0] == 0 && color.val[1] == 150 && color.val[2] == 0){
                    newColor[0] = 160; newColor[1] = 160; newColor[2] = 160; //180
                }
                else if(color.val[0] == 0 && color.val[1] == 200 && color.val[2] == 0){
                    newColor[0] = 110; newColor[1] = 110; newColor[2] = 110; //180
                }
                else if(color.val[0] == 0 && color.val[1] == 250 && color.val[2] == 0){
                    newColor[0] = 255; newColor[1] = 255; newColor[2] = 255; //180
                }
                end1801.at<Vec3b>(i, j) = newColor;
            }
        }

        for(int i = 0; i < img2701.rows; i++)
        {
            for(int j = 0; j < img2701.cols; j++)
            {
                Vec3b color = img2701.at<Vec3b>(i, j);
                Vec3b newColor;
                if(color.val[0] == 160 && color.val[1] == 160 && color.val[2] == 160){
                    newColor[0] = 0; newColor[1] = 50; newColor[2] = 0;
                }
                else if(color.val[0] == 110 && color.val[1] == 110 && color.val[2] == 110){
                    newColor[0] = 0; newColor[1] = 100; newColor[2] = 0;
                }
                else if(color.val[0] == 60 && color.val[1] == 60 && color.val[2] == 60){
                    newColor[0] = 0; newColor[1] = 150; newColor[2] = 0;
                }
                else if(color.val[0] == 210 && color.val[1] == 210 && color.val[2] == 210){
                    newColor[0] = 0; newColor[1] = 200; newColor[2] = 0;
                }
                else if(color.val[0] == 255 && color.val[1] == 255 && color.val[2] == 255){
                    newColor[0] = 0; newColor[1] = 250; newColor[2] = 0;
                }

                temp2701.at<Vec3b>(i, j) = newColor;
            }
        }

        for(int i = 0; i < temp2701.rows; i++)
        {
            for(int j = 0; j < temp2701.cols; j++)
            {
                Vec3b color = temp2701.at<Vec3b>(i, j);
                Vec3b newColor;
                if(color.val[0] == 0 && color.val[1] == 50 && color.val[2] == 0){
                    newColor[0] = 210; newColor[1] = 210; newColor[2] = 210; //270
                }
                else if(color.val[0] == 0 && color.val[1] == 100 && color.val[2] == 0){
                    newColor[0] = 160; newColor[1] = 160; newColor[2] = 160; //270
                }
                else if(color.val[0] == 0 && color.val[1] == 150 && color.val[2] == 0){
                    newColor[0] = 110; newColor[1] = 110; newColor[2] = 110; //270
                }
                else if(color.val[0] == 0 && color.val[1] == 200 && color.val[2] == 0){
                    newColor[0] = 60; newColor[1] = 60; newColor[2] = 60; //270
                }
                else if(color.val[0] == 0 && color.val[1] == 250 && color.val[2] == 0){
                    newColor[0] = 255; newColor[1] = 255; newColor[2] = 255; //270
                }
                end2701.at<Vec3b>(i, j) = newColor;
            }
        }

        imwrite("200\\MaskeRotirane\\" + to_string(i+160) + ".png", end901);
        imwrite("200\\MaskeRotirane\\" + to_string(i+320) + ".png", end1801);
        imwrite("200\\MaskeRotirane\\" + to_string(i+480) + ".png", end2701);

        //Slagalice

        img902 = imread("200\\SlagaliceRotirane90\\" + to_string(i) + ".png");
        img1802 = imread("200\\SlagaliceRotirane180\\" + to_string(i) + ".png");
        img2702 = imread("200\\SlagaliceRotirane270\\" + to_string(i) + ".png");

        imwrite("200\\SlagaliceRotirane\\" + to_string(i+160) + ".png", img902);
        imwrite("200\\SlagaliceRotirane\\" + to_string(i+320) + ".png", img1802);
        imwrite("200\\SlagaliceRotirane\\" + to_string(i+480) + ".png", img2702);
    }
}

int imageCounter = 1;

class Puzzle{
    Mat image;
    vector <Point> corners; //upLeft, upRight, downRight, downLeft;
    vector <Point> sidePoints; //up, right, down, left
    vector <int> sideInOrOut; //0-udubljenje, 1-ispupčenje, -1-ravno
    vector <Mat> sides; //up, right, down, left
    vector <vector<double>> threeDistancesEverySide; //tri distance po svakoj strani

public:
    Puzzle(String path){
        image = imread(path, IMREAD_COLOR);

        findCorners();
        sidePoints.push_back(inOrOutUp());
        sidePoints.push_back(inOrOutRight());
        sidePoints.push_back(inOrOutDown());
        sidePoints.push_back(inOrOutLeft());

        setThreeDistancesEverySide();
        findSidesForTemplates();
    }

    Mat getImage(){
        return image;
    }

    vector <Point> getSpecificColorPixels(Mat img){
        vector<Point> pixels;
        findNonZero(img, pixels);
        return pixels;
    }

    void findCorners(){
        /*!
        ########################################################
            Traženje kornera se zasniva na traženju piksela specifične boje sa slika kontura koje su prethodno formirane
        ########################################################
        */

        Mat upLeftCorner;
        inRange(getImage(), Scalar(0, 60, 0), Scalar(0, 60, 0), upLeftCorner);
        vector <Point> upLeftPixel = getSpecificColorPixels(upLeftCorner);
        corners.push_back(upLeftPixel.at(0));

        Mat upRightCorner;
        inRange(getImage(), Scalar(0, 210, 0), Scalar(0, 210, 0), upRightCorner);
        vector <Point> upRightPixel = getSpecificColorPixels(upRightCorner);
        corners.push_back(upRightPixel.at(0));

        Mat downRightCorner;
        inRange(getImage(), Scalar(0, 160, 0), Scalar(0, 160, 0), downRightCorner);
        vector <Point> downRightPixel = getSpecificColorPixels(downRightCorner);
        corners.push_back(downRightPixel.at(0));

        Mat downLeftCorner;
        inRange(getImage(), Scalar(0, 110, 0), Scalar(0, 110, 0), downLeftCorner);
        vector <Point> downLeftPixel = getSpecificColorPixels(downLeftCorner);
        corners.push_back(downLeftPixel.at(0));
    }

    vector <Point> getCorners(){
        return corners;
    }

    Point inOrOutUp(){
        /*!
        ########################################################
            Da li gornja ivica ima ispupčenje ili udubljenje.
            Provjeravaju se sve tačke gornje konture, te se određuje maksimalna i minimalna y koordinata.
            Na osnovu njih i na osnovu ćoškova gornje konture se određuje ispupčenje/udubljenje.
            Također, funkcija vraća tačku vrha ispupčenja/udubljenja.
        ########################################################
        */
        Mat redContour; //gornja ivica
        inRange(getImage(), Scalar(0, 0, 150), Scalar(0, 0, 150), redContour);
        vector <Point> redPixels = getSpecificColorPixels(redContour);

        Point maxPoint, minPoint;
        int maxYCoord = 0, minYCoord = 10000;

        for(uint i = 0; i < redPixels.size(); i++){
            if(redPixels.at(i).y > maxYCoord)
                maxYCoord = redPixels.at(i).y;
            if(redPixels.at(i).y < minYCoord)
                minYCoord = redPixels.at(i).y;
        }

        int xCoord = 0, yCoord = 0;
        int counter = 0; //brojač koordinata koje imaju istu y koordinatu, a ujedno su max ili min

        if(abs(maxYCoord - (getCorners().at(0).y + getCorners().at(1).y)/2) > abs(minYCoord - (getCorners().at(0).y + getCorners().at(1).y)/2)){
            sideInOrOut.push_back(0);
            yCoord = maxYCoord;
            for(uint i = 0; i < redPixels.size(); i++){
                if(redPixels.at(i).y == maxYCoord){
                    xCoord += redPixels.at(i).x;
                    counter ++;
                }
            }
        }
        else{
            sideInOrOut.push_back(1);
            yCoord = minYCoord;
            for(uint i = 0; i < redPixels.size(); i++){
                if(redPixels.at(i).y == minYCoord){
                    xCoord += redPixels.at(i).x;
                    counter ++;
                }
            }
        }
        xCoord = xCoord/counter;

        if(abs(yCoord - (getCorners().at(0).y + getCorners().at(1).y)/2) < 20){
            sideInOrOut.pop_back();
            sideInOrOut.push_back(-1);
        }

        return Point(xCoord, yCoord);
    }

    Point inOrOutRight(){
        /*!
        ########################################################
            Da li desna ivica ima ispupčenje ili udubljenje.
            Provjeravaju se sve tačke desne konture, te se određuje maksimalna i minimalna x koordinata.
            Na osnovu njih i na osnovu ćoškova desne konture se određuje ispupčenje/udubljenje.
            Također, funkcija vraća tačku vrha ispupčenja/udubljenja.
        ########################################################
        */
        Mat blueContour; //desna ivica
        inRange(getImage(), Scalar(150, 0, 0), Scalar(150, 0, 0), blueContour);
        vector <Point> bluePixels = getSpecificColorPixels(blueContour);

        Point maxPoint, minPoint;
        int maxXCoord = 0, minXCoord = 10000;

        for(uint i = 0; i < bluePixels.size(); i++){
            if(bluePixels.at(i).x > maxXCoord)
                maxXCoord = bluePixels.at(i).x;
            if(bluePixels.at(i).x < minXCoord)
                minXCoord = bluePixels.at(i).x;
        }

        int xCoord = 0, yCoord = 0;
        int counter = 0; //brojač koordinata koje imaju istu y koordinatu, a ujedno su max ili min

        if(abs(maxXCoord - (getCorners().at(1).x + getCorners().at(2).x)/2) > abs(minXCoord - (getCorners().at(1).x + getCorners().at(2).x)/2)){
            sideInOrOut.push_back(1);
            xCoord = maxXCoord;
            for(uint i = 0; i < bluePixels.size(); i++){
                if(bluePixels.at(i).x == maxXCoord){
                    yCoord += bluePixels.at(i).y;
                    counter ++;
                }
            }
        }
        else{
            sideInOrOut.push_back(0);
            xCoord = minXCoord;
            for(uint i = 0; i < bluePixels.size(); i++){
                if(bluePixels.at(i).x == minXCoord){
                    yCoord += bluePixels.at(i).y;
                    counter ++;
                }
            }
        }
        yCoord = yCoord/counter;

        if(abs(xCoord - (getCorners().at(1).x + getCorners().at(2).x)/2) < 20){
            sideInOrOut.pop_back();
            sideInOrOut.push_back(-1);
        }

        return Point(xCoord, yCoord);
    }

    Point inOrOutDown(){
        /*!
        ########################################################
            Da li donja ivica ima ispupčenje ili udubljenje.
            Provjeravaju se sve tačke donje konture, te se određuje maksimalna i minimalna y koordinata.
            Na osnovu njih i na osnovu ćoškova donje konture se određuje ispupčenje/udubljenje.
            Također, funkcija vraća tačku vrha ispupčenja/udubljenja.
        ########################################################
        */
        Mat pinkContour; //donja ivica
        inRange(getImage(), Scalar(150, 0, 150), Scalar(150, 0, 150), pinkContour);
        vector <Point> greenPixels = getSpecificColorPixels(pinkContour);

        Point maxPoint, minPoint;
        int maxYCoord = 0, minYCoord = 10000;

        for(uint i = 0; i < greenPixels.size(); i++){
            if(greenPixels.at(i).y > maxYCoord)
                maxYCoord = greenPixels.at(i).y;
            if(greenPixels.at(i).y < minYCoord)
                minYCoord = greenPixels.at(i).y;
        }

        int xCoord = 0, yCoord = 0;
        int counter = 0; //brojač koordinata koje imaju istu y koordinatu, a ujedno su max ili min

        if(abs(maxYCoord - (getCorners().at(2).y + getCorners().at(3).y)/2) > abs(minYCoord - (getCorners().at(2).y + getCorners().at(3).y)/2)){
            sideInOrOut.push_back(1);
            yCoord = maxYCoord;
            for(uint i = 0; i < greenPixels.size(); i++){
                if(greenPixels.at(i).y == maxYCoord){
                    xCoord += greenPixels.at(i).x;
                    counter ++;
                }
            }
        }
        else{
            sideInOrOut.push_back(0);
            yCoord = minYCoord;
            for(uint i = 0; i < greenPixels.size(); i++){
                if(greenPixels.at(i).y == minYCoord){
                    xCoord += greenPixels.at(i).x;
                    counter ++;
                }
            }
        }
        xCoord = xCoord/counter;

        if(abs(yCoord - (getCorners().at(2).y + getCorners().at(3).y)/2) < 20){
            sideInOrOut.pop_back();
            sideInOrOut.push_back(-1);
        }

        return Point(xCoord, yCoord);
    }

    Point inOrOutLeft(){
        /*!
        ########################################################
            Da li lijeva ivica ima ispupčenje ili udubljenje.
            Provjeravaju se sve tačke lijeve konture, te se određuje maksimalna i minimalna x koordinata.
            Na osnovu njih i na osnovu ćoškova lijeve konture se određuje ispupčenje/udubljenje.
            Također, funkcija vraća tačku vrha ispupčenja/udubljenja.
        ########################################################
        */
        Mat greenContour; //lijeva ivica
        inRange(getImage(), Scalar(0, 150, 0), Scalar(0, 150, 0), greenContour);
        vector <Point> pinkPixels = getSpecificColorPixels(greenContour);

        Point maxPoint, minPoint;
        int maxXCoord = 0, minXCoord = 10000;

        for(uint i = 0; i < pinkPixels.size(); i++){
            if(pinkPixels.at(i).x > maxXCoord)
                maxXCoord = pinkPixels.at(i).x;
            if(pinkPixels.at(i).x < minXCoord)
                minXCoord = pinkPixels.at(i).x;
        }

        int xCoord = 0, yCoord = 0;
        int counter = 0; //brojač koordinata koje imaju istu y koordinatu, a ujedno su max ili min

        if(abs(maxXCoord - (getCorners().at(3).x + getCorners().at(0).x)/2) > abs(minXCoord - (getCorners().at(3).x + getCorners().at(0).x)/2)){
            sideInOrOut.push_back(0);
            xCoord = maxXCoord;
            for(uint i = 0; i < pinkPixels.size(); i++){
                if(pinkPixels.at(i).x == maxXCoord){
                    yCoord += pinkPixels.at(i).y;
                    counter ++;
                }
            }
        }
        else{
            sideInOrOut.push_back(1);
            xCoord = minXCoord;
            for(uint i = 0; i < pinkPixels.size(); i++){
                if(pinkPixels.at(i).x == minXCoord){
                    yCoord += pinkPixels.at(i).y;
                    counter ++;
                }
            }
        }
        yCoord = yCoord/counter;

        if(abs(xCoord - (getCorners().at(3).x + getCorners().at(0).x)/2) < 20){
            sideInOrOut.pop_back();
            sideInOrOut.push_back(-1);
        }

        return Point(xCoord, yCoord);
    }

    vector <Point> getSidePoints(){
        return sidePoints;
    }

    vector <int> getSideInOrOut(){
        return sideInOrOut;
    }

    void setThreeDistancesEverySide(){
        /*!
        ########################################################
            Funkcija za određivanje po tri dužine svake strane puzle.
            Prva dužina: između dva susjedna ćoška jedne od strana
            Druga dužina: između jednog ćoška i vršne tačke ispupčenja/udubljenja
            Treća dužina: između drugog ćoška i vršne tačke ispupčenja/udubljenja
        ########################################################
        */
        vector<double> tempUp, tempRight, tempDown, tempLeft;
        tempUp.push_back(norm(getCorners().at(0) - getCorners().at(1)));
        tempUp.push_back(norm(getCorners().at(0) - getSidePoints().at(0)));
        tempUp.push_back(norm(getCorners().at(1) - getSidePoints().at(0)));
        threeDistancesEverySide.push_back(tempUp);

        tempRight.push_back(norm(getCorners().at(1) - getCorners().at(2)));
        tempRight.push_back(norm(getCorners().at(1) - getSidePoints().at(1)));
        tempRight.push_back(norm(getCorners().at(2) - getSidePoints().at(1)));
        threeDistancesEverySide.push_back(tempRight);

        tempDown.push_back(norm(getCorners().at(3) - getCorners().at(2)));
        tempDown.push_back(norm(getCorners().at(3) - getSidePoints().at(2)));
        tempDown.push_back(norm(getCorners().at(2) - getSidePoints().at(2)));
        threeDistancesEverySide.push_back(tempDown);

        tempLeft.push_back(norm(getCorners().at(0) - getCorners().at(3)));
        tempLeft.push_back(norm(getCorners().at(0) - getSidePoints().at(3)));
        tempLeft.push_back(norm(getCorners().at(3) - getSidePoints().at(3)));
        threeDistancesEverySide.push_back(tempLeft);
    }

    vector <vector<double>> getThreeDistancesEverySide(){
        return threeDistancesEverySide;
    }

    void findSidesForTemplates(){
        /*!
        ########################################################
            Izdvajanje strana puzli u zasebne slike koje će kasnije služiti u template matchingu
        ########################################################
        */
        int xMin, yMin, xMax, yMax;
        int shift = 25;

        Mat maskUp;
        inRange(getImage(), Scalar(0, 0, 150), Scalar(0, 0, 150), maskUp);

        xMin = min({getCorners().at(0).x, getCorners().at(1).x, getSidePoints().at(0).x}) - shift;
        yMin = min({getCorners().at(0).y, getCorners().at(1).y, getSidePoints().at(0).y}) - shift;
        xMax = max({getCorners().at(0).x, getCorners().at(1).x, getSidePoints().at(0).x}) + shift;
        yMax = max({getCorners().at(0).y, getCorners().at(1).y, getSidePoints().at(0).y}) + shift;
        maskUp(Rect(xMin,yMin,xMax-xMin,yMax-yMin)).copyTo(maskUp);

        Mat maskRight;
        inRange(getImage(), Scalar(150, 0, 0), Scalar(150, 0, 0), maskRight);

        xMin = min({getCorners().at(1).x, getCorners().at(2).x, getSidePoints().at(1).x}) - shift;
        yMin = min({getCorners().at(1).y, getCorners().at(2).y, getSidePoints().at(1).y}) - shift;
        xMax = max({getCorners().at(1).x, getCorners().at(2).x, getSidePoints().at(1).x}) + shift;
        yMax = max({getCorners().at(1).y, getCorners().at(2).y, getSidePoints().at(1).y}) + shift;
        maskRight(Rect(xMin,yMin,xMax-xMin,yMax-yMin)).copyTo(maskRight);

        Mat maskDown;
        inRange(getImage(), Scalar(150, 0, 150), Scalar(150, 0, 150), maskDown);

        xMin = min({getCorners().at(2).x, getCorners().at(3).x, getSidePoints().at(2).x}) - shift;
        yMin = min({getCorners().at(2).y, getCorners().at(3).y, getSidePoints().at(2).y}) - shift;
        xMax = max({getCorners().at(2).x, getCorners().at(3).x, getSidePoints().at(2).x}) + shift;
        yMax = max({getCorners().at(2).y, getCorners().at(3).y, getSidePoints().at(2).y}) + shift;
        maskDown(Rect(xMin,yMin,xMax-xMin,yMax-yMin)).copyTo(maskDown);

        Mat maskLeft;
        inRange(getImage(), Scalar(0, 150, 0), Scalar(0, 150, 0), maskLeft);

        xMin = min({getCorners().at(0).x, getCorners().at(3).x, getSidePoints().at(3).x}) - shift;
        yMin = min({getCorners().at(0).y, getCorners().at(3).y, getSidePoints().at(3).y}) - shift;
        xMax = max({getCorners().at(0).x, getCorners().at(3).x, getSidePoints().at(3).x}) + shift;
        yMax = max({getCorners().at(0).y, getCorners().at(3).y, getSidePoints().at(3).y}) + shift;
        maskLeft(Rect(xMin,yMin,xMax-xMin,yMax-yMin)).copyTo(maskLeft);

        sides.push_back(maskUp);
        sides.push_back(maskRight);
        sides.push_back(maskDown);
        sides.push_back(maskLeft);
    }

    vector <Mat> getSides(){
        return sides;
    }
};

int numberOfPuzzles = 160;
int numRowsOfEndPuzzle = 16;
int numColsOfEndPuzzle = 10;

int numberOfPuzzleRotation = 4; // uvijek bi ovaj broj trebao biti 4, jer svaku puzlu posmatramo na način da se ona može uklopiti sa 4 strane
vector <Puzzle> puzzle;
vector<int> numbersOfPuzzles; // 1 2 3 4 5 6 7 ....., i postavljam ih na 0 kada im nađem mjesto u konačnoj slici
vector<vector<int>> endPuzzle(numRowsOfEndPuzzle, vector<int> (numColsOfEndPuzzle)); // matrica konačno složene puzle

void ucitajPuzleUKlasu(){ // učitavanje svih puzli (kontura debljine 5) u klasu
    for(int i = 1; i < numberOfPuzzleRotation*numberOfPuzzles + 1; i++){
        string pom = "200\\Konture\\" + to_string(i) + ".png";
        puzzle.push_back(Puzzle(pom));
        imageCounter ++;
    }
}

vector <int> findSidesOfNonZeroNeighbor(vector<vector<int>> v){ //funkcija za traženje susjeda puzle čije mjesto je već nađeno
    vector <int> temp;
    for(int i = 0; i < int(v.size()); i++){
        for(int j = 0; j < int(v.at(0).size()); j++){
            if(v.at(i).at(j) == 0){
                temp.push_back(i);
                temp.push_back(j);

                if(i + 1 < numRowsOfEndPuzzle && v.at(i+1).at(j) != 0){
                    temp.push_back(v.at(i+1).at(j)-1);
                    temp.push_back(0);
                }
                if(j - 1 >= 0 && v.at(i).at(j-1) != 0){
                    temp.push_back(v.at(i).at(j-1)-1);
                    temp.push_back(1);
                }
                if(i - 1 >= 0 && v.at(i-1).at(j) != 0){
                    temp.push_back(v.at(i-1).at(j)-1);
                    temp.push_back(2);
                }
                if(j + 1 < numColsOfEndPuzzle && v.at(i).at(j+1) != 0){
                    temp.push_back(v.at(i).at(j+1)-1);
                    temp.push_back(3);
                }
                if(temp.size() != 2)
                    return temp;
                else{
                    temp.pop_back();
                    temp.pop_back();
                }
            }
        }
    }
    return temp;
}

/*vector <int> findSidesOfNonZeroNeighborRight(vector<vector<int>> v){ //funkcija za traženje susjeda puzle čije mjesto je već nađeno
    vector <int> temp;
    for(int i = 0; i < int(v.size()); i++){
        for(int j = int(v.at(0).size()) - 1; j >= 0; j--){
            if(v.at(i).at(j) == 0){
                temp.push_back(i);
                temp.push_back(j);

                if(i + 1 < numRowsOfEndPuzzle && v.at(i+1).at(j) != 0){
                    temp.push_back(v.at(i+1).at(j)-1);
                    temp.push_back(0);
                }
                if(j - 1 >= 0 && v.at(i).at(j-1) != 0){
                    temp.push_back(v.at(i).at(j-1)-1);
                    temp.push_back(1);
                }
                if(i - 1 >= 0 && v.at(i-1).at(j) != 0){
                    temp.push_back(v.at(i-1).at(j)-1);
                    temp.push_back(2);
                }
                if(j + 1 < numColsOfEndPuzzle && v.at(i).at(j+1) != 0){
                    temp.push_back(v.at(i).at(j+1)-1);
                    temp.push_back(3);
                }
                if(temp.size() != 2)
                    return temp;
                else{
                    temp.pop_back();
                    temp.pop_back();
                }
            }
        }
    }
    return temp;
}*/

double checkTemplate(Mat image1, Mat image2){ //ispitivanje template matchinga

    int offset1 = 10;
    int offset2 = 60;
    copyMakeBorder(image1, image1, offset2, offset2, offset2, offset2, BORDER_CONSTANT);
    copyMakeBorder(image2, image2, offset1, offset1, offset1, offset1, BORDER_CONSTANT);

    Mat temp = image1;

    Point2f center((image2.cols - 1) / 2.0, (image2.rows - 1) / 2.0);
    double maxCoef = 0;
    int flag = 0;

    for (int k = -10; k < 10; k += 1){

        Mat M = getRotationMatrix2D(center, double(k), 1.0);
        Mat rotirana;
        warpAffine(image2, rotirana, M, image2.size());

        Mat rez;
        matchTemplate(rotirana, temp, rez, TM_CCOEFF_NORMED);
        threshold(rotirana, rotirana, 100, 255, THRESH_BINARY);

        double minVal; double maxVal; Point minLoc; Point maxLoc;
        minMaxLoc( rez, &minVal, &maxVal, &minLoc, &maxLoc);

        if(flag == 1 && maxVal < maxCoef) //ako koefijent počne opadati odmah vrati maksimalni koeficijent
            return maxCoef;
        if(maxCoef > 0.75) //ako je koefijent prešao 0.75 tada se radi o veoma dobrom poklapanju, te ga je potrebno nekako zabilježiti
            flag = 1;
        if (maxVal > maxCoef) //ukoliko je novi koeficijent veći od prethodnog, prethodni = novi
            maxCoef = maxVal;
        if(maxVal < 0.4) //ukoliko je koefijent manji od 0.4 povećaj korak za 3 (ubrzanje algoritma)
            k = k + 3;
        if(maxVal < 0.5) //ukoliko je koefijent manji od 0.5 povećaj korak za 2 (ubrzanje algoritma)
            k = k + 2;
    }

    return maxCoef;
}

double puzzleMatchingCoefficient(vector <int> v, int midlePuzzle){

    //Provjera da li se nalazimo uz ivicu konačne slike puzle
    if(v.at(0) == 0 && puzzle.at(midlePuzzle-1).getSideInOrOut().at(0) != -1) //gornja ivica
        return 1000;
    if(v.at(0) == numRowsOfEndPuzzle-1 && puzzle.at(midlePuzzle-1).getSideInOrOut().at(2) != -1) //donja ivica
        return 10000;
    if(v.at(1) == numColsOfEndPuzzle-1 && puzzle.at(midlePuzzle-1).getSideInOrOut().at(1) != -1) //desna ivica
        return 100000;
    if(v.at(1) == 0 && puzzle.at(midlePuzzle-1).getSideInOrOut().at(3) != -1) //lijeva ivica
        return 1000000;

    int counter = 0;

    for(uint i = 1; i < v.size()/2; i++){
        // da li se "srednja puzla" može "ući" u susjedne
        if(puzzle.at(v.at(i*2)).getSideInOrOut().at(v.at(2*i + 1)) + puzzle.at(midlePuzzle-1).getSideInOrOut().at((v.at(2*i + 1) + 2)%4) == 1)
            counter ++;
    }

    // ukoliko "srednja puzla" ne može ući u sve susjedne odmah vrati neki veliki broj
    if(counter != v.size()/2 - 1) return 20000;

    double threeDistancesResult = 0; //razlika između tri distance strana puzli koje bi se trebale uklapati jedna u drugu
    for(uint i = 1; i < v.size()/2; i++){
        for(int j = 0; j < 3; j++){
            threeDistancesResult = threeDistancesResult + abs(puzzle.at(v.at(i*2)).getThreeDistancesEverySide().at(v.at(2*i + 1)).at(j) - puzzle.at(midlePuzzle-1).getThreeDistancesEverySide().at((v.at(2*i + 1)+2)%4).at(j));
        }
    }

    if(threeDistancesResult > 55) //ako je ova razlika prevelika puzle sigurno ne odgovaraju jedna drugoj
        return 9000;

    double tempResult = 0; //rezultat template matchinga
    for(uint i = 1; i < v.size()/2; i++){
        tempResult = tempResult + checkTemplate(puzzle.at(v.at(i*2)).getSides().at(v.at(2*i + 1)), puzzle.at(midlePuzzle-1).getSides().at((v.at(2*i + 1) + 2)%4));
    }

    return (v.size()/2 - 1)-tempResult;
}

pair<Point2f, Point2f> dajTrecuTacku(Point2f A, Point2f B){
    double a = norm(A - B);
    Point2f M = (A + B) / 2;
    double k1 = (B.y - A.y)/(B.x - A.x);
    if(k1 == 0)
        k1 = 0.1;
    double k2 = -1/k1;

    double h = a*sqrt(3) / 2;

    double x1 = (2 * M.x + 2*k2*k2 * M.x +
            sqrt( (2*M.x + 2*k2*k2 * M.x) * (2*M.x + 2*k2*k2 * M.x) - 4*(1 + k2*k2) * (M.x * M.x + k2 * k2 * M.x * M.x - h*h) ) ) /
            (2*(1 + k2*k2));

    double x2 = (2 * M.x + 2*k2*k2 * M.x -
            sqrt( (2*M.x + 2*k2*k2 * M.x) * (2*M.x + 2*k2*k2 * M.x) - 4*(1 + k2*k2) * (M.x * M.x + k2 * k2 * M.x * M.x - h*h) ) ) /
            (2*(1 + k2*k2));

    double n = M.y - k2*M.x;
    double y1 = k2 * x1 + n;
    double y2 = k2 * x2 + n;

    Point2f C1(x1, y1);
    Point2f C2(x2, y2);

    return pair(C1, C2);
}

double odrediPoklapanje(vector<double> prvi, vector<double> drugi){
    double vrati = 100000;
    int oduzmi = 0;
    //for(int oduzmi = -2; oduzmi < 3; oduzmi=oduzmi + 2){

        if(prvi.size() > drugi.size()){
            for(int i = 0; i < prvi.size() - drugi.size(); i++){
                double suma = 0;

                for(int j = 0; j < drugi.size(); j++){
                    double modifikovaniDrugi = drugi[j] + oduzmi;
                    suma = suma + abs(prvi[i+j] - modifikovaniDrugi);
                }
                suma = suma / drugi.size();
                if(suma < vrati) vrati = suma;
            }

        }

        if(prvi.size() < drugi.size()){
            for(int i = 0; i < drugi.size() - prvi.size(); i++){
                double suma = 0;

                for(int j = 0; j < prvi.size(); j++){
                    double modifikovaniPrvi = prvi[j] + oduzmi;
                    suma = suma + abs(drugi[i+j] - modifikovaniPrvi);
                }
                suma = suma / prvi.size();
                if(suma < vrati) vrati = suma;
            }
        }
        if(prvi.size() == drugi.size()) {
            return 2.1;
        }
    //}
    return vrati;
}

vector< tuple<bool, vector <vector<double> >, vector <vector<double> > > > dajSvePodatke (){
    /**
    -ova struktura ima boj redova onoliko koliko ima slagalica, to je prvi vektor
    -zatim za svaku od slagalica traži unutrašnju ili vanjsku konturu. Unutrasnju konuturu
    za slagalicu za koju se trazi slagalica koja ce se uklopiti, a vanjsku za slagalicu
    koja ce se u nju uklopiti
    -sljedeci vektor je sve cetriti strane slagalice
    -i konacno sve tacke na svakoj od strana slagalice.
    **/
    vector< tuple<bool, vector <vector<double> >, vector <vector<double> > > > SlagalicaSviPodaci;
    int im = 1;
    while(1) {
        String path = "200\\KontureDebljina1\\" + to_string(im) + ".png";
        Mat slika = imread(path);
        if(slika.empty()) {
            break;
        }
        Mat slikaKontura = Mat::zeros( slika.size(), CV_8UC3 );

        Point cosakGL, cosakDL, cosakDD, cosakGD;

        vector<Point> coskovi;

        for(int i = 0; i < slika.rows; i++){
            for(int j = 0; j < slika.cols; j++){
                uchar B = slika.at<Vec3b>(i, j)[0];
                uchar G = slika.at<Vec3b>(i, j)[1];
                uchar R = slika.at<Vec3b>(i, j)[2];

                Point tacka(j, i);

                if(G == 60 && R == 0 && B == 0)
                    cosakGL = Point(j, i);

                if(G == 110 && R == 0 && B == 0)
                    cosakDL = Point(j, i);

                if(G == 160 && R == 0 && B == 0)
                    cosakDD = Point(j, i);

                if(G == 210 && R == 0 && B == 0)
                    cosakGD = Point(j, i);
            }
        }

        cvtColor(slika,slika, COLOR_BGR2GRAY);

        vector<vector<Point> > konture;
        findContours( slika, konture, RETR_TREE, CHAIN_APPROX_NONE );
        //drawContours( slikaKontura, konture, -1, Scalar(255,0,0));


        coskovi.push_back(cosakGL);
        coskovi.push_back(cosakDL);
        coskovi.push_back(cosakDD);
        coskovi.push_back(cosakGD);

        vector<Point> gore1;
        vector<Point> lijevo1;
        vector<Point> dole1;
        vector<Point> desno1;
        vector<Point> goreOstatak1;
        vector<Point> gorePravi1;

        int orjenatacija = 1;

        for( size_t z = 0; z < konture[0].size(); z++ ) {
            for(size_t h = 0; h < coskovi.size(); h++){
                if(konture[0][z] == coskovi[h]) {
                    orjenatacija++;
                }
            }
            if(orjenatacija == 1)
                gore1.push_back(konture[0][z]);
            if(orjenatacija == 2)
                lijevo1.push_back(konture[0][z]);
            if(orjenatacija == 3)
                dole1.push_back(konture[0][z]);
            if(orjenatacija == 4)
                desno1.push_back(konture[0][z]);
            if(orjenatacija == 5)
                goreOstatak1.push_back(konture[0][z]);
        }

        for( size_t p = 0; p < goreOstatak1.size(); p++) {
            gorePravi1.push_back(goreOstatak1[p]);
        }

        for( size_t p = 0; p < gore1.size(); p++) {
            gorePravi1.push_back(gore1[p]);
        }
        int ofset = 2;

        vector<Point> lijevo(lijevo1.begin() + ofset, lijevo1.begin() + lijevo1.size() - 2*ofset);
        vector<Point> dole2(dole1.begin() + ofset, dole1.begin() + dole1.size() - 2*ofset);
        vector<Point> desno2(desno1.begin() + ofset, desno1.begin() + desno1.size() - 2*ofset);
        vector<Point> gorePravi(gorePravi1.begin() + ofset, gorePravi1.begin() + gorePravi1.size() - 2*ofset);
        vector<Point> dole;
        for(int i = 0; i < dole2.size(); i++){
            dole.push_back(dole2[dole2.size() - 1 - i]);
        }

        vector<Point> desno;
        for(int i = 0; i < desno2.size(); i++){
            desno.push_back(desno2[desno2.size() - 1 - i]);
        }

        vector<double> norma_gore_unutra;
        vector<double> norma_gore_vani;

        vector<double> norma_desno_unutra;
        vector<double> norma_desno_vani;

        vector<double> norma_dole_unutra;
        vector<double> norma_dole_vani;

        vector<double> norma_lijevo_unutra;
        vector<double> norma_lijevo_vani;

        ///GORE
        pair treceTackeGore = dajTrecuTacku((gorePravi[0] + gorePravi[1]) / 2, (gorePravi[gorePravi.size()-1] + gorePravi[gorePravi.size()-2]) / 2);
        Point2f srednjaTackaGore = (gorePravi[0] + gorePravi[gorePravi.size()-1]) / 2;
        Point vani, unutra;

        if(treceTackeGore.first.y < srednjaTackaGore.y) {
            //VANI
            vani = treceTackeGore.first;
            unutra = treceTackeGore.second;
        }

        else {
            //UNUTRA
            unutra = treceTackeGore.first;
            vani = treceTackeGore.second;
        }

        //Size si(500, 300);
        //Mat drawing1 = Mat::zeros( si, CV_8UC3 );
        //Mat drawing2 = Mat::zeros( si, CV_8UC3 );

        for( size_t p = 0; p < gorePravi.size(); p++) {
            norma_gore_unutra.push_back(norm(unutra - gorePravi[p]));
            norma_gore_vani.push_back(norm(vani - gorePravi[p]));

            //Point tackaUnutra(p+1, norm(unutra - gorePravi[p]));
            //Point tackaVani(p+1, norm(vani - gorePravi[p]));

            //circle(drawing1, tackaUnutra, 1, Scalar(0, 255, 0), -1);
            //circle(drawing2, tackaVani, 1, Scalar(0, 255, 0), -1);
        }

        ///LIJEVO
        pair treceTackeLijevo = dajTrecuTacku((lijevo[0] + lijevo[1]) / 2, (lijevo[lijevo.size() - 1] + lijevo[lijevo.size() - 2])/2);
        Point2f srednjaTackaLijevo = (lijevo[0] + lijevo[lijevo.size() - 1]) / 2;
        if(treceTackeLijevo.first.x < srednjaTackaLijevo.x) {
            vani = treceTackeLijevo.first;
            unutra = treceTackeLijevo.second;
        }
        else {
            unutra = treceTackeLijevo.first;
            vani = treceTackeLijevo.second;
        }
        for( size_t p = 0; p < lijevo.size(); p++) {
            norma_lijevo_unutra.push_back(norm(unutra - lijevo[p]));
            norma_lijevo_vani.push_back(norm(vani - lijevo[p]));
        }


        ///DOLE
        pair treceTackeDole = dajTrecuTacku((dole[0] + dole[1])/2, (dole[dole.size()-1] + dole[dole.size()-1])/2);
        Point2f srednjaTackaDole = (dole[0] + dole[dole.size()-1]) / 2;
        if(treceTackeDole.first.y > srednjaTackaDole.y) {
            vani = treceTackeDole.first;
            unutra = treceTackeDole.second;
        }
        else {
            unutra = treceTackeDole.first;
            vani = treceTackeDole.second;
        }
        for( size_t p = 0; p < dole.size(); p++) {
            norma_dole_unutra.push_back(norm(unutra - dole[p]));
            norma_dole_vani.push_back(norm(vani - dole[p]));
        }


        ///DESNO
        pair treceTackeDesno = dajTrecuTacku((desno[0] + desno[1])/2, (desno[desno.size()-1] + desno[desno.size()-1])/2);
        Point2f srednjaTackaDesno = (desno[0] + desno[desno.size()-1]) / 2;
        if(treceTackeDesno.first.x < srednjaTackaDesno.x) {
            unutra = treceTackeDesno.first;
            vani = treceTackeDesno.second;
        }
        else {
            vani = treceTackeDesno.first;
            unutra = treceTackeDesno.second;
        }

        for( size_t p = 0; p < desno.size(); p++) {
            norma_desno_unutra.push_back(norm(unutra - desno[p]));
            norma_desno_vani.push_back(norm(vani - desno[p]));
        }

        tuple<bool, vector <vector<double> >, vector <vector<double> > > privremeniTuple;

        vector <vector<double> > prvi;
        vector <vector<double> > drugi;

        prvi.push_back(norma_gore_unutra);
        prvi.push_back(norma_lijevo_unutra);
        prvi.push_back(norma_dole_unutra);
        prvi.push_back(norma_desno_unutra);

        drugi.push_back(norma_gore_vani);
        drugi.push_back(norma_lijevo_vani);
        drugi.push_back(norma_dole_vani);
        drugi.push_back(norma_desno_vani);

        get<0>(privremeniTuple) = false;
        get<1>(privremeniTuple) = prvi;
        get<2>(privremeniTuple) = drugi;

        SlagalicaSviPodaci.push_back(privremeniTuple);

        im++;
    }
    return SlagalicaSviPodaci;

}

void rotirajSlagalicaSviPodaci(int brojSlagalice, int ugao){

    /*!
    ########################################################
        Ukoliko je izvornu slagalicu potrebno zarotirati da bi se ona ispravno uklopila,
        potrebno je u nastavku posmatrati njene zarotirane dužine formirane u funkciji 'dajSvePodatke'
    ########################################################
    */
    vector<double> unutraSlagalicaGore = get<1>(SlagaliceSviPodaci[brojSlagalice])[(0 + ugao)%4];
    vector<double> unutraSlagalicaLijevo = get<1>(SlagaliceSviPodaci[brojSlagalice])[(1 + ugao)%4];
    vector<double> unutraSlagalicaDole = get<1>(SlagaliceSviPodaci[brojSlagalice])[(2 + ugao)%4];
    vector<double> unutraSlagalicaDesno = get<1>(SlagaliceSviPodaci[brojSlagalice])[(3 + ugao)%4];

    get<1>(SlagaliceSviPodaci[brojSlagalice])[0] = unutraSlagalicaGore;
    get<1>(SlagaliceSviPodaci[brojSlagalice])[1] = unutraSlagalicaLijevo;
    get<1>(SlagaliceSviPodaci[brojSlagalice])[2] = unutraSlagalicaDole;
    get<1>(SlagaliceSviPodaci[brojSlagalice])[3] = unutraSlagalicaDesno;

    vector<double> vaniSlagalicaGore = get<2>(SlagaliceSviPodaci[brojSlagalice])[(0 + ugao)%4];
    vector<double> vaniSlagalicaLijevo = get<2>(SlagaliceSviPodaci[brojSlagalice])[(1 + ugao)%4];
    vector<double> vaniSlagalicaDole = get<2>(SlagaliceSviPodaci[brojSlagalice])[(2 + ugao)%4];
    vector<double> vaniSlagalicaDesno = get<2>(SlagaliceSviPodaci[brojSlagalice])[(3 + ugao)%4];

    get<2>(SlagaliceSviPodaci[brojSlagalice])[0] = vaniSlagalicaGore;
    get<2>(SlagaliceSviPodaci[brojSlagalice])[1] = vaniSlagalicaLijevo;
    get<2>(SlagaliceSviPodaci[brojSlagalice])[2] = vaniSlagalicaDole;
    get<2>(SlagaliceSviPodaci[brojSlagalice])[3] = vaniSlagalicaDesno;
}

vector <pair<int, string>> dajNajbolje(int brojSlagalice, String smijer){

    int gore = 0, lijevo = 1, dole = 2,desno = 3;

    vector<double> prva;
    if(smijer == "desno")
        prva = get<1>(SlagaliceSviPodaci[brojSlagalice - 1])[desno];
    if(smijer == "gore")
        prva = get<1>(SlagaliceSviPodaci[brojSlagalice - 1])[gore];
    if(smijer == "lijevo")
        prva = get<1>(SlagaliceSviPodaci[brojSlagalice - 1])[lijevo];
    if(smijer == "dole")
        prva = get<1>(SlagaliceSviPodaci[brojSlagalice - 1])[dole];

    vector <pair <int, string> > vrati;

    for(int i = 0; i < 160 ; i++){
        vector<double> slagalicaLijevo = get<2>(SlagaliceSviPodaci[i])[lijevo];
        vector<double> slagalicaDesno = get<2>(SlagaliceSviPodaci[i])[desno];
        vector<double> slagalicaGore = get<2>(SlagaliceSviPodaci[i])[gore];
        vector<double> slagalicaDole = get<2>(SlagaliceSviPodaci[i])[dole];

        //cout << slagalicaLijevo.size() << " +++++" << endl;

        double poklapanjeLijevo = odrediPoklapanje(prva, slagalicaLijevo);
        double poklapanjeDesno = odrediPoklapanje(prva, slagalicaDesno);
        double poklapanjeGore = odrediPoklapanje(prva, slagalicaGore);
        double poklapanjeDole = odrediPoklapanje(prva, slagalicaDole);

        if(abs( int(prva.size()) - int(slagalicaLijevo.size())) < 35) {
            if (poklapanjeLijevo < 20) {
                String stranicaSlagalice = to_string(i + 1) + "_lijevo";
                pair<int, String> par;
                par.first = i + 1;//poklapanjeLijevo;
                par.second = "lijevo";//stranicaSlagalice;
                vrati.push_back(par);
                  //cout << stranicaSlagalice << " "<< poklapanjeLijevo<<endl;
            }
        }
        if(abs( int(prva.size()) - int(slagalicaDesno.size())) < 35) {
            if (poklapanjeDesno < 20) {
                String stranicaSlagalice = to_string(i + 1) + "_desno";
                pair<int, String> par;
                par.first = i + 1;//poklapanjeDesno;
                par.second = "desno";//stranicaSlagalice;
                vrati.push_back(par);
                    //cout << stranicaSlagalice << " "<<poklapanjeDesno<< endl;
            }
        }
        if(abs( int(prva.size()) - int(slagalicaGore.size())) < 35) {
            if (poklapanjeGore < 20) {
                String stranicaSlagalice = to_string(i + 1) + "_gore";
                pair<int, String> par;
                par.first = i + 1;//poklapanjeGore;
                par.second = "gore";//stranicaSlagalice;
                vrati.push_back(par);
                    //cout << stranicaSlagalice << " "<<poklapanjeGore<<endl;
            }
        }
        if(abs( int(prva.size()) - int(slagalicaDole.size())) < 35) {
            if (poklapanjeDole < 20) {
                //cout << "USOO";
                String stranicaSlagalice = to_string(i + 1) + "_dole";
                pair<int, String> par;
                //cout << "#" << poklapanjeDole << " " << i+1 << "#"<<endl;
                par.first = i + 1;//poklapanjeDole;
                par.second = "dole";//stranicaSlagalice;
                //cout << par.first;
                vrati.push_back(par);
                    //cout << stranicaSlagalice << " "<<poklapanjeDole<<endl;
            }
        }
    }
/*
    sort(vrati.begin(), vrati.end());
*/
    //for(int i = 0; i < vrati.size(); i++){
        //cout << vrati[i].first << " " << vrati[i].second<<endl;
    //}
    //cout << vrati.size() << "************" << endl;

    return vrati;

}
///vratiće broj slagalice koja se moze uklopiti, stranu dole, pa stranu lijevo

vector <tuple<int, String, String>> dajNajbolje(int brojPrveSlagalice, int brojDrugeSlagalice, int smjer){
    vector <pair<int, String>> uklapajPrvaStrana, uklapajDrugaStrana;
    if(smjer == 1){
        uklapajPrvaStrana = dajNajbolje(brojPrveSlagalice, "desno");
        uklapajDrugaStrana = dajNajbolje(brojDrugeSlagalice, "dole");
    }
    else{
        uklapajPrvaStrana = dajNajbolje(brojPrveSlagalice, "gore");
        uklapajDrugaStrana = dajNajbolje(brojDrugeSlagalice, "desno");
    }

    vector <tuple<int, String, String>> vrati;

    for(uint i = 0; i < uklapajPrvaStrana.size(); i++){
        for(uint j = 0; j < uklapajDrugaStrana.size(); j++){
            if(uklapajPrvaStrana[i].first == uklapajDrugaStrana[j].first){

                if(     (uklapajPrvaStrana[i].second == "dole" && uklapajDrugaStrana[j].second == "lijevo") ||
                        (uklapajPrvaStrana[i].second == "desno" && uklapajDrugaStrana[j].second == "dole") ||
                        (uklapajPrvaStrana[i].second == "gore" && uklapajDrugaStrana[j].second == "desno") ||
                        (uklapajPrvaStrana[i].second == "lijevo" && uklapajDrugaStrana[j].second == "gore") )

                    vrati.push_back(make_tuple(uklapajPrvaStrana[i].first, uklapajPrvaStrana[i].second, uklapajDrugaStrana[j].second));
            }
        }
    }

    return vrati;
}

Point2f rotacija2D(const Point2f& inPoint, const double& angRad){
    Point2f rotiranaTacka;
    rotiranaTacka.x = cos(angRad)*inPoint.x - sin(angRad)*inPoint.y;
    rotiranaTacka.y = sin(angRad)*inPoint.x + cos(angRad)*inPoint.y;
    return rotiranaTacka;
}

Point2f rotirajTacku(const cv::Point2f& inPoint, const cv::Point2f& center, const double& angRad){
    return rotacija2D(inPoint - center, -angRad) + center;
}

int pomjeriLijevo = 0;
vector<int> pomjeriGore(numColsOfEndPuzzle);
int rasporedi = 0;
int dodatno = 0;
vector<int> vectorPomjeriLijevo;

pair<vector<int>, vector<int> > sloziNarednu(bool uso2, int red, int kol, vector<int> podaciPrethonaLijevo,
                                             vector<int> podaciPrethonaGore, int brojSlagalice){
    pair<vector<int>, vector<int> >vrati;
    vector<int> desno;
    vector<int> dole;

    String path = "200//MaskeRotirane//" + to_string(brojSlagalice) + ".png";
    Mat slagalicaMaska = imread(path, IMREAD_GRAYSCALE);

    String path1 = "200//SlagaliceRotirane//" + to_string(brojSlagalice) + ".png";
    Mat slagalica = imread(path1);

    int udaljenostGLLijevo, udaljenostGLGore, udaljenostDLLijevo, udaljenostDLDole;
    int udaljenostGDDesno, udaljenostGDGore, udaljenostDDDesno, udaljenostDDDole;

    int horizontalnaRazlikaLijevo, vertikalnaRazlikaLijevo;
    int horizontalnaRazlikaGore, vertikalnaRazlikaGore;

    Point tackaGL, tackaDL, tackaGD, tackaDD;
    for(int i = 0; i < slagalicaMaska.rows; i++) {
        for(int j = 0; j <  slagalicaMaska.cols; j++) {
            double piksel =  slagalicaMaska.at<uchar>(i, j);
            if(piksel == 60){
                udaljenostGLLijevo = j;
                udaljenostGLGore = i;
                tackaGL = Point(j, i);
            }
            if(piksel == 110){
                udaljenostDLLijevo = j;
                udaljenostDLDole = velicinaKvadrata - i;
                tackaDL = Point(j, i);
            }
            if(piksel == 160){
                udaljenostDDDesno = velicinaKvadrata - j;
                udaljenostDDDole = velicinaKvadrata - i;
                tackaDD = Point(j, i);
            }
            if(piksel == 210){
                udaljenostGDDesno = velicinaKvadrata - j;
                udaljenostGDGore = i;
                tackaGD = Point(j, i);
            }
        }
    }

    if(red == 0){
        podaciPrethonaGore[0] = udaljenostGLLijevo;
        podaciPrethonaGore[1] = 0;
        podaciPrethonaGore[2] = udaljenostGDDesno;
        podaciPrethonaGore[3] = 0;
    }

    if(kol == 0){
        podaciPrethonaLijevo[0] = 0;
        podaciPrethonaLijevo[1] = udaljenostGLGore;
        podaciPrethonaLijevo[2] = 0;
        podaciPrethonaLijevo[3] = udaljenostDLDole;
    }

    if(kol == 0 && red != 0){
        podaciPrethonaLijevo[0] = 0;
        podaciPrethonaLijevo[1] = podaciPrethonaGore[1];
        podaciPrethonaLijevo[2] = 0;
        podaciPrethonaLijevo[3] = podaciPrethonaGore[3];
    }

    /// Na kraju treba biti paralelogram.............


    int konstantaHorizontalno = podaciPrethonaLijevo[0] + udaljenostGLLijevo;
    int konstantaVertikalno = podaciPrethonaGore[1] + udaljenostGLGore;

    //cout << konstantaVertikalno << " ";
    Mat rotation_matix;
    double ugaoRotacije;
    if(kol != 9) {
        horizontalnaRazlikaLijevo = konstantaHorizontalno - podaciPrethonaLijevo[2] - udaljenostDLLijevo;
        vertikalnaRazlikaLijevo = velicinaKvadrata - udaljenostGLGore - udaljenostDLDole;
        double ugaoRotacijeLijevo = atan(double(horizontalnaRazlikaLijevo)/double(vertikalnaRazlikaLijevo) );

        vertikalnaRazlikaGore = konstantaVertikalno - podaciPrethonaGore[3] - udaljenostGDGore;
        horizontalnaRazlikaGore = velicinaKvadrata - udaljenostGLLijevo - udaljenostGDDesno;
        double ugaoRotacijeGore = -atan(double(vertikalnaRazlikaGore) / double(horizontalnaRazlikaGore));

        ugaoRotacije = (ugaoRotacijeLijevo + ugaoRotacijeGore) / 2;
        rotation_matix = getRotationMatrix2D(tackaGL, ugaoRotacije * 180./CV_PI, 1.0);
    }
    else{
        int horizontalnaRazlikaDesno = udaljenostGDDesno - udaljenostDDDesno;
        int vertikalnaRazlikaDesno = velicinaKvadrata - udaljenostGLGore - udaljenostDLDole;
        ugaoRotacije = -atan(double( horizontalnaRazlikaDesno)/double(vertikalnaRazlikaDesno) );

        rotation_matix = getRotationMatrix2D(tackaGD, ugaoRotacije * 180./CV_PI, 1.0);
        konstantaVertikalno = podaciPrethonaGore[3] + udaljenostGDGore;
    }
    //cout << konstantaVertikalno << endl;


    warpAffine(slagalicaMaska, slagalicaMaska, rotation_matix, slagalicaMaska.size());
    warpAffine(slagalica, slagalica, rotation_matix, slagalica.size());

    threshold(slagalicaMaska, slagalicaMaska, 100, 255, 0);



    float translacijaVerikalno = 0;
    float translacijaHorizontalno = 0;

    translacijaHorizontalno = (podaciPrethonaGore[0] - udaljenostGLLijevo);
    translacijaVerikalno = podaciPrethonaLijevo[1] - udaljenostGLGore;

    if(kol == 0){
        translacijaVerikalno = 0;
        //translacijaHorizontalno = podaciPrethonaGore[0] - udaljenostGLLijevo;
    }

    if(red == 0){
        translacijaHorizontalno = 0;
        //translacijaVerikalno = podaciPrethonaLijevo[1] - udaljenostGLGore;
    }


    float warp_values1[] = { 1.0, 0.0, 0, 0.0, 1.0, translacijaVerikalno};
    Mat translation_matrix1 = Mat(2, 3, CV_32F, warp_values1);

    float warp_values2[] = { 1.0, 0.0, translacijaHorizontalno, 0.0, 1.0, 0};
    Mat translation_matrix2 = Mat(2, 3, CV_32F, warp_values2);

    warpAffine(slagalicaMaska, slagalicaMaska, translation_matrix1, slagalicaMaska.size());
    warpAffine(slagalica, slagalica, translation_matrix1, slagalica.size());

    warpAffine(slagalicaMaska, slagalicaMaska, translation_matrix2, slagalicaMaska.size());
    warpAffine(slagalica, slagalica, translation_matrix2, slagalica.size());


    Point tackaGD1;
    Point tackaGL1;
    Point tackaDD1;
    Point tackaDL1;

    if(kol != 9){
        tackaGD1.x = rotirajTacku(tackaGD, tackaGL, ugaoRotacije).x + translacijaHorizontalno;
        tackaGD1.y = rotirajTacku(tackaGD, tackaGL, ugaoRotacije).y + translacijaVerikalno;

        tackaDD1.x = rotirajTacku(tackaDD, tackaGL, ugaoRotacije).x + translacijaHorizontalno;
        tackaDD1.y = rotirajTacku(tackaDD, tackaGL, ugaoRotacije).y + translacijaVerikalno;

        tackaGL1.x = tackaGL.x + translacijaHorizontalno;
        tackaGL1.y = tackaGL.y + translacijaVerikalno;

        tackaDL1.x = rotirajTacku(tackaDL, tackaGL, ugaoRotacije).x + translacijaHorizontalno;
        tackaDL1.y = rotirajTacku(tackaDL, tackaGL, ugaoRotacije).y + translacijaVerikalno;
    }


    if(kol == numColsOfEndPuzzle - 1){
        tackaDD1.x = rotirajTacku(tackaDD, tackaGD, ugaoRotacije).x + translacijaHorizontalno;
        tackaDD1.y = rotirajTacku(tackaDD, tackaGD, ugaoRotacije).y + translacijaVerikalno;

        tackaGL1.x = rotirajTacku(tackaGL, tackaGD, ugaoRotacije).x + translacijaHorizontalno;
        tackaGL1.y = rotirajTacku(tackaGL, tackaGD, ugaoRotacije).y + translacijaVerikalno;

        tackaGD1.x = tackaGD.x + translacijaHorizontalno;
        tackaGD1.y = tackaGD.y + translacijaVerikalno;

        tackaDL1.x = rotirajTacku(tackaDL, tackaGD, ugaoRotacije).x + translacijaHorizontalno;
        tackaDL1.y = rotirajTacku(tackaDL, tackaGD, ugaoRotacije).y + translacijaVerikalno;
    }

    /*
    circle(slagalicaMaska, tackaGL1, 0, Scalar(60), -1);
    circle(slagalicaMaska, tackaDL1, 0, Scalar(110), -1);
    circle(slagalicaMaska, tackaDD1, 0, Scalar(160), -1);
    circle(slagalicaMaska, tackaGD1, 0, Scalar(210), -1);
*/
    for(int i = 0; i < slagalicaMaska.rows; i++) {
        for(int j = 0; j <  slagalicaMaska.cols; j++) {
            double piksel =  slagalicaMaska.at<uchar>(i, j);

            if(piksel == 210){
                slagalicaMaska.at<uchar>(i, j) = 255;
                int pomjerioSe = 0;
                for(int l = 1; l < 10; l++){
                    if(slagalicaMaska.at<uchar>(i - l , j) != 0){
                        pomjerioSe++;
                        tackaGD1 = Point(j, i - l);
                    }
                }
                i = i - pomjerioSe;

                pomjerioSe = 0;
                for(int l = 1; l < 10; l++){
                    if(slagalicaMaska.at<uchar>(i , j + l) != 0){
                        pomjerioSe++;
                        tackaGD1 = Point(j + l, i);
                    }
                }
                j = j + pomjerioSe;

                int pomjerioSeDesno = 1;
                for(int k = 0; k < 6; k++) {
                    for(int l = pomjerioSeDesno; l < 10; l++) {
                        if(slagalicaMaska.at<uchar>(i + k , j + l) != 0){
                            tackaGD1 = Point(j + l, i + k);
                        }
                        else {
                            pomjerioSeDesno = l+1;
                            break;
                        }
                    }
                }
            }

            if(piksel == 60){
                slagalicaMaska.at<uchar>(i, j) = 255;

                int pomjerioSe = 0;
                for(int l = 1; l < 10; l++){
                    if(slagalicaMaska.at<uchar>(i - l , j) != 0){
                        pomjerioSe++;
                        tackaGD1 = Point(j, i - l);
                    }
                }
                i = i - pomjerioSe;

                pomjerioSe = 0;
                for(int l = 1; l < 10; l++){
                    if(slagalicaMaska.at<uchar>(i , j - l) != 0){
                        pomjerioSe++;
                        tackaGD1 = Point(j - l, i);
                    }
                }
                j = j - pomjerioSe;

                int pomjerioSeLijevo = 1;
                for(int k = 0; k < 6; k++) {
                    for(int l = pomjerioSeLijevo; l < 10; l++) {
                        if(slagalicaMaska.at<uchar>(i + k , j - l) != 0){
                            tackaGL1 = Point(j - l, i + k);
                        }
                        else {
                            pomjerioSeLijevo = l+1;
                            break;
                        }

                    }
                }
            }

            if(piksel == 160){
                slagalicaMaska.at<uchar>(i, j) = 255;

                int pomjerioSe = 0;
                for(int l = 1; l < 10; l++){
                    if(slagalicaMaska.at<uchar>(i + l , j) != 0){
                        pomjerioSe++;
                        tackaDD1 = Point(j, i + l);
                    }
                }
                i = i + pomjerioSe;

                pomjerioSe = 0;
                for(int l = 1; l < 10; l++){
                    if(slagalicaMaska.at<uchar>(i , j + l) != 0){
                        pomjerioSe++;
                        tackaDD1 = Point(j + l, i);
                    }
                }
                j = j + pomjerioSe;

                int pomjerioSeDesno = 1;
                for(int k = 0; k < 6; k++) {
                    for(int l = pomjerioSeDesno; l < 10; l++) {
                        if(slagalicaMaska.at<uchar>(i - k , j + l) != 0){
                            tackaDD1 = Point(j + l, i - k);
                        }
                        else {
                            pomjerioSeDesno = l+1;
                            break;
                        }
                    }
                }
            }

            if(piksel == 110){
                slagalicaMaska.at<uchar>(i, j) = 255;
                int pomjerioSe = 0;
                for(int l = 1; l < 10; l++){
                    if(slagalicaMaska.at<uchar>(i + l , j) != 0){
                        pomjerioSe++;
                        tackaDL1 = Point(j, i + l);
                    }
                }
                i = i + pomjerioSe;

                pomjerioSe = 0;
                for(int l = 1; l < 10; l++){
                    if(slagalicaMaska.at<uchar>(i , j - l) != 0){
                        pomjerioSe++;
                        tackaDL1 = Point(j - l, i);
                    }
                }
                j = j - pomjerioSe;

                int pomjerioSeLijevo = 1;
                for(int k = 0; k < 6; k++) {
                    for(int l = pomjerioSeLijevo; l < 10; l++) {
                        if(slagalicaMaska.at<uchar>(i - k , j - l) != 0){
                            tackaDL1 = Point(j - l, i - k);
                        }
                        else {
                            pomjerioSeLijevo = l+1;
                            break;
                        }
                    }
                }
            }
        }
    }

    /*
    circle(slagalicaMaska, tackaGL1,0, Scalar(150), -1);
    circle(slagalicaMaska, tackaGD1,0, Scalar(150), -1);
    circle(slagalicaMaska, tackaDL1,0, Scalar(150), -1);
    circle(slagalicaMaska, tackaDD1,0, Scalar(150), -1);
    */

    udaljenostGDDesno = velicinaKvadrata - tackaGD1.x;
    udaljenostGDGore = tackaGD1.y;

    udaljenostDDDesno = velicinaKvadrata - tackaDD1.x;
    udaljenostDDDole = velicinaKvadrata - tackaDD1.y;

    udaljenostDLLijevo = tackaDL1.x;
    udaljenostDLDole = velicinaKvadrata - tackaDL1.y;

    udaljenostGLLijevo = tackaGL1.x;
    udaljenostGLGore = tackaGL1.y;

    konstantaHorizontalno = ( (podaciPrethonaLijevo[0] + udaljenostGLLijevo) + (podaciPrethonaLijevo[2] + udaljenostDLLijevo) ) / 2;
    konstantaVertikalno = ( (podaciPrethonaGore[1] + udaljenostGLGore) + (podaciPrethonaGore[3] + udaljenostGDGore) ) / 2;


    desno.push_back(udaljenostGDDesno);
    desno.push_back(udaljenostGDGore);
    desno.push_back(udaljenostDDDesno);
    desno.push_back(udaljenostDDDole);
    //desno.push_back(udaljenostGLLijevo);

    dole.push_back(udaljenostDLLijevo);
    dole.push_back(udaljenostDLDole);
    dole.push_back(udaljenostDDDesno);
    dole.push_back(udaljenostDDDole);

    vrati.first = desno;
    vrati.second = dole;

    if(kol == 0)
        pomjeriLijevo = 0;
    else
        pomjeriLijevo = pomjeriLijevo + konstantaHorizontalno;

    if(kol ==9)
        vectorPomjeriLijevo.push_back(pomjeriLijevo);

    if(red == 0)
        pomjeriGore[kol] = 0;
    else
        pomjeriGore[kol] = pomjeriGore[kol] + konstantaVertikalno - 7;

    if(uso2 == true){


        if(kol == 0){
             rasporedi = vectorPomjeriLijevo[0] - vectorPomjeriLijevo[red];
             //cout <<endl<< rasporedi<<endl;
        }
        int dodatnoUOvomKrugu = rasporedi / (numColsOfEndPuzzle - kol);
        rasporedi = rasporedi - dodatnoUOvomKrugu;

       //cout << dodatnoUOvomKrugu;
        pomjeriLijevo = pomjeriLijevo + dodatnoUOvomKrugu - 7;

        for(int i = 0; i < slagalica.rows; i++){
            for(int j = 0; j < slagalica.cols; j++){
                if(slagalicaMaska.at<uchar>(i, j) > 200) {
                    KonacnaSlagalica.at<Vec3b>(velicinaKvadrata * red + i - pomjeriGore[kol], velicinaKvadrata * kol + j - pomjeriLijevo )[0] = slagalica.at<Vec3b>(i, j)[0];
                    KonacnaSlagalica.at<Vec3b>(velicinaKvadrata * red + i - pomjeriGore[kol], velicinaKvadrata * kol + j - pomjeriLijevo )[1] = slagalica.at<Vec3b>(i, j)[1];
                    KonacnaSlagalica.at<Vec3b>(velicinaKvadrata * red + i - pomjeriGore[kol], velicinaKvadrata * kol + j - pomjeriLijevo )[2] = slagalica.at<Vec3b>(i, j)[2];
                }
            }
        }

        //oVideoWriter.write(KonacnaSlagalica);
    }

    return vrati;
}

vector<vector<int>> algoritamZaSlaganjePuzli(){
    for(int i = 1; i < numberOfPuzzleRotation*numberOfPuzzles + 1; i++)
        numbersOfPuzzles.push_back(i);

    //traženje početne puzle
    int smjer = 0; //smjer slaganja puzli, 1 - odozgo, 0 - odozdo

    for(int i = 0; i < numberOfPuzzleRotation*numberOfPuzzles; i++){
        if(puzzle.at(i).getSideInOrOut().at(0) == -1 && puzzle.at(i).getSideInOrOut().at(3) == -1){
            endPuzzle.at(0).at(0) = numbersOfPuzzles.at(i%numberOfPuzzles);

            //znak da smo složeli određenu puzlu, te je u narednim koracima slaganja nije potrebno uopšte posmatrati
            numbersOfPuzzles.at(i) = 0;
            numbersOfPuzzles.at(numberOfPuzzles + i%numberOfPuzzles) = 0;
            numbersOfPuzzles.at(2*numberOfPuzzles + i%numberOfPuzzles) = 0;
            numbersOfPuzzles.at(3*numberOfPuzzles + i%numberOfPuzzles) = 0;

            rotirajSlagalicaSviPodaci(i%numberOfPuzzles, int(i/numberOfPuzzles));
            get<0>(SlagaliceSviPodaci[i%numberOfPuzzles]) = true;

            //gornji lijevi ćošak, znači puzla će se slagati odozgo
            smjer = 1;
            break;
        }
        if(puzzle.at(i).getSideInOrOut().at(0) == -1 && puzzle.at(i).getSideInOrOut().at(1) == -1){
            endPuzzle.at(0).at(numColsOfEndPuzzle-1) = numbersOfPuzzles.at(i%numberOfPuzzles);

            //znak da smo složeli određenu puzlu, te je u narednim koracima slaganja nije potrebno uopšte posmatrati
            numbersOfPuzzles.at(i%numberOfPuzzles) = 0;
            numbersOfPuzzles.at(numberOfPuzzles + i%numberOfPuzzles) = 0;
            numbersOfPuzzles.at(2*numberOfPuzzles + i%numberOfPuzzles) = 0;
            numbersOfPuzzles.at(3*numberOfPuzzles + i%numberOfPuzzles) = 0;

            rotirajSlagalicaSviPodaci(i%numberOfPuzzles, int(i/numberOfPuzzles));
            get<0>(SlagaliceSviPodaci[i%numberOfPuzzles]) = true;

            //gornji desni ćošak, znači puzla će se slagati odozgo
            smjer = 1;
            break;
        }
        if(puzzle.at(i).getSideInOrOut().at(2) == -1 && puzzle.at(i).getSideInOrOut().at(3) == -1){
            endPuzzle.at(numRowsOfEndPuzzle-1).at(0) = numbersOfPuzzles.at(i%numberOfPuzzles);

            //znak da smo složeli određenu puzlu, te je u narednim koracima slaganja nije potrebno uopšte posmatrati
            numbersOfPuzzles.at(i%numberOfPuzzles) = 0;
            numbersOfPuzzles.at(numberOfPuzzles + i%numberOfPuzzles) = 0;
            numbersOfPuzzles.at(2*numberOfPuzzles + i%numberOfPuzzles) = 0;
            numbersOfPuzzles.at(3*numberOfPuzzles + i%numberOfPuzzles) = 0;

            rotirajSlagalicaSviPodaci(i%numberOfPuzzles, int(i/numberOfPuzzles));
            get<0>(SlagaliceSviPodaci[i%numberOfPuzzles]) = true;

            //donji lijevi ćošak, znači puzla će se slagati odozdo
            smjer = 0;
            break;
        }
        if(puzzle.at(i).getSideInOrOut().at(1) == -1 && puzzle.at(i).getSideInOrOut().at(2) == -1){
            endPuzzle.at(numRowsOfEndPuzzle-1).at(numColsOfEndPuzzle-1) = numbersOfPuzzles.at(i%numberOfPuzzles);

            //znak da smo složeli određenu puzlu, te je u narednim koracima slaganja nije potrebno uopšte posmatrati
            numbersOfPuzzles.at(i%numberOfPuzzles) = 0;
            numbersOfPuzzles.at(numberOfPuzzles + i%numberOfPuzzles) = 0;
            numbersOfPuzzles.at(2*numberOfPuzzles + i%numberOfPuzzles) = 0;
            numbersOfPuzzles.at(3*numberOfPuzzles + i%numberOfPuzzles) = 0;

            rotirajSlagalicaSviPodaci(i%numberOfPuzzles, int(i/numberOfPuzzles));
            get<0>(SlagaliceSviPodaci[i%numberOfPuzzles]) = true;

            //donji desni ćošak, znači puzla će se slagati odozdo
            smjer = 0;
            break;
        }
    }


    int counter = 0;
    int desno = 0; //desno = 0 - sa lijeve strane; desno = 1 - sa desne strane

    while(counter != numberOfPuzzles-1){//kako smo našli početnu puzlu, ostalo ih je još ukupan broj puzli - 1

        vector <int> neighbors;
        if(desno == 0) //ako će se počevši od drugog reda slagalica slagati sa lijeve strane
            neighbors = findSidesOfNonZeroNeighbor(endPuzzle);
        /*else //ako će se počevši od drugog reda slagalica slagati sa desne strane
            neighbors = findSidesOfNonZeroNeighborRight(endPuzzle);*/

        vector <double> coefs; //koefijenti uklapanja svake puzle sa trenutnom/im koje su složene

        if(neighbors.size() == 4){ //ako slažemo prvi/posljednji red/kolonu, tada za trenutnu puzlu imamo samo jednog susjeda
            for(int i = 0; i < numberOfPuzzleRotation*numberOfPuzzles; i++){
                if(numbersOfPuzzles.at(i) != 0){ //ako puzla već nije pronađena u konačnoj slici
                    coefs.push_back(puzzleMatchingCoefficient(neighbors, numbersOfPuzzles.at(i)));
                }
                else{
                    coefs.push_back(5000);
                }
            }
        }
        else{
            vector <tuple<int, String, String>> possiblePuzzles = dajNajbolje(neighbors.at(2)%numberOfPuzzles + 1 , neighbors.at(4)%numberOfPuzzles + 1, smjer);
            for(int i = 0; i < numberOfPuzzles*numberOfPuzzleRotation; i++){
                int flag = 0;
                for(uint j = 0; j < possiblePuzzles.size(); j++){ //među svim puzlama, izdvojene su potencijalne koje bi se mogle uklapati
                    if((i == get<0>(possiblePuzzles[j])-1 && get<1>(possiblePuzzles[j]) == "lijevo" && get<2>(possiblePuzzles[j]) == "gore") ||
                            (i == numberOfPuzzles + get<0>(possiblePuzzles[j])-1 && get<1>(possiblePuzzles[j]) == "gore" && get<2>(possiblePuzzles[j]) == "desno") ||
                            (i == 2*numberOfPuzzles + get<0>(possiblePuzzles[j])-1 && get<1>(possiblePuzzles[j]) == "desno" && get<2>(possiblePuzzles[j]) == "dole") ||
                            (i == 3*numberOfPuzzles + get<0>(possiblePuzzles[j])-1 && get<1>(possiblePuzzles[j]) == "dole" && get<2>(possiblePuzzles[j]) == "lijevo")){
                        flag = 1;
                        if(numbersOfPuzzles.at(i) != 0){
                            coefs.push_back(puzzleMatchingCoefficient(neighbors, numbersOfPuzzles.at(i)));
                        }
                        else{
                            coefs.push_back(6000);
                        }
                        break;
                    }
                }
                if(flag == 0)
                    coefs.push_back(7000);
            }
        }
        //u konačnu matricu upisujemo index najmanjeg elementa vektora koeficijenata
        endPuzzle.at(neighbors.at(0)).at(neighbors.at(1)) = numbersOfPuzzles.at(min_element(coefs.begin(), coefs.end()) - coefs.begin());

        //znak da smo složeli određenu puzlu, te je u narednim koracima slaganja nije potrebno uopšte posmatrati
        numbersOfPuzzles.at((min_element(coefs.begin(), coefs.end()) - coefs.begin())%numberOfPuzzles) = 0;
        numbersOfPuzzles.at(numberOfPuzzles + (min_element(coefs.begin(), coefs.end()) - coefs.begin())%numberOfPuzzles) = 0;
        numbersOfPuzzles.at(2*numberOfPuzzles + (min_element(coefs.begin(), coefs.end()) - coefs.begin())%numberOfPuzzles) = 0;
        numbersOfPuzzles.at(3*numberOfPuzzles + (min_element(coefs.begin(), coefs.end()) - coefs.begin())%numberOfPuzzles) = 0;

        rotirajSlagalicaSviPodaci((min_element(coefs.begin(), coefs.end()) - coefs.begin())%numberOfPuzzles, int((min_element(coefs.begin(), coefs.end()) - coefs.begin())/numberOfPuzzles));
        get<0>(SlagaliceSviPodaci[numbersOfPuzzles.at(min_element(coefs.begin(), coefs.end()) - coefs.begin()) - 1]) = true;

        counter++;
    }

    for (int i = 0; i < numRowsOfEndPuzzle; i++){
        for(int j = 0; j < numColsOfEndPuzzle; j++){
            cout << setw(4) << (endPuzzle.at(i).at(j)-1)%numberOfPuzzles + 1;
        }
        cout << endl;
    }

    return endPuzzle;
}

void stvaranjeKonacneSlike(){

    vector<vector<int>> prethodniGore;
    pair<vector<int>, vector<int>> prethodna;
    for (int i = 0; i < numRowsOfEndPuzzle; i++){
        for(int j = 0; j < numColsOfEndPuzzle; j++){
            if(i == 0 && j == 0){
                vector<int> prethodnaGore(4);
                vector<int> prethodnaLijevo(4);

                prethodna = sloziNarednu(false,i, j, prethodnaLijevo, prethodnaGore, endPuzzle.at(i).at(j));
                prethodniGore.push_back(prethodna.second);
            }
            else if(i == 0 && j != 0){
                vector<int> prethodnaGore(4);
                prethodna = sloziNarednu(false,i, j, prethodna.first, prethodnaGore, endPuzzle.at(i).at(j));
                prethodniGore.push_back(prethodna.second);
            }
            else{
                prethodna = sloziNarednu(false,i, j, prethodna.first, prethodniGore[j], endPuzzle.at(i).at(j));
                prethodniGore[j] = prethodna.second;
            }
        }
    }

    vector<vector<int>> prethodniGore1;
    pair<vector<int>, vector<int>> prethodna1;
    for (int i = 0; i < numRowsOfEndPuzzle; i++){
        for(int j = 0; j < numColsOfEndPuzzle; j++){
            if(i == 0 && j == 0){
                vector<int> prethodnaGore1(4);
                vector<int> prethodnaLijevo1(4);

                prethodna1 = sloziNarednu(true, i, j, prethodnaLijevo1, prethodnaGore1, endPuzzle.at(i).at(j));
                prethodniGore1.push_back(prethodna1.second);
            }
            else if(i == 0){
                vector<int> prethodnaGore1(4);
                prethodna1 = sloziNarednu(true, i, j, prethodna1.first, prethodnaGore1, endPuzzle.at(i).at(j));
                prethodniGore1.push_back(prethodna1.second);
            }
            else{
                prethodna1 = sloziNarednu(true, i, j, prethodna1.first, prethodniGore1[j], endPuzzle.at(i).at(j));
                prethodniGore1[j] = prethodna1.second;
            }
        }
    }
}

int main() {
    clock_t start, end;
    start = clock();

    cout << "Ucitavanje i predobrada slika..." << endl;
    mkdir("200//KontureDebljina1");
    mkdir("200//KontureDebljina5");
    mkdir("200//Slagalice");
    mkdir("200//Maske");

    izdvojiSlagalice(1);
    izdvojiSlagalice(5);
    dodajBorderKontureMaskeSlagalice();
    dodajSveRotacijeKontureMaskeSlagalice();

    SlagaliceSviPodaci = dajSvePodatke();
    ucitajPuzleUKlasu();
    end = clock();
    double vrijeme_akvizicija = double(end - start) / double(CLOCKS_PER_SEC);
        cout << "Utroseno vrijeme za akviziciju slagalica: " << fixed
             << vrijeme_akvizicija << setprecision(5) << "s" << endl << endl;


    //oVideoWriter.open("slaganjeSlagalice.mp4", VideoWriter::fourcc('m','p','4','v'), 5, KonacnaSlagalica.size(), true);
    cout << "Slaganje slagalica pokrenuto..." << endl;
    start = clock();
    algoritamZaSlaganjePuzli();

    end = clock();
    double vrijeme_slaganja_matrice = double(end - start) / double(CLOCKS_PER_SEC);
        cout << "Vrijeme formiranja matrice konacne slagalice: " << fixed
             << vrijeme_slaganja_matrice << setprecision(5) << "s" << endl << endl;

    start = clock();
    stvaranjeKonacneSlike();
    end = clock();
    double vrijeme_izrade_konacne_slike = double(end - start) / double(CLOCKS_PER_SEC);
        cout << "Vrijeme formiranja konacne slike puzle: " << fixed
             << vrijeme_izrade_konacne_slike << setprecision(5) << "s" << endl;
    //oVideoWriter.release();


    filesystem::remove_all("200\\Konture0");
    filesystem::remove_all("200\\Konture90");
    filesystem::remove_all("200\\Konture180");
    filesystem::remove_all("200\\Konture270");

    filesystem::remove_all("200\\SlagaliceRotirane0");
    filesystem::remove_all("200\\SlagaliceRotirane90");
    filesystem::remove_all("200\\SlagaliceRotirane180");
    filesystem::remove_all("200\\SlagaliceRotirane270");

    filesystem::remove_all("200\\MaskeRotirane0");
    filesystem::remove_all("200\\MaskeRotirane90");
    filesystem::remove_all("200\\MaskeRotirane180");
    filesystem::remove_all("200\\MaskeRotirane270");

    filesystem::remove_all("200\\Konture");
    filesystem::remove_all("200\\KontureDebljina1");
    filesystem::remove_all("200\\Maske");
    filesystem::remove_all("200\\MaskeRotirane");
    filesystem::remove_all("200\\SlagaliceRotirane");
    filesystem::remove_all("200\\Slagalice");


    imwrite("200\\SlozenaSlika.png", KonacnaSlagalica);

    namedWindow ("PrikazSlagalice", WINDOW_NORMAL);
    imshow("PrikazSlagalice", KonacnaSlagalica);
    waitKey();
    return 0;
}
