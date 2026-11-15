use std::io::Cursor;
use weavetab_gate::{Gate, Input};

fn main() {
    let gate = Gate::default();

    if let Some(img_path) = std::env::args().nth(1) {
        let bytes = std::fs::read(&img_path).expect("failed to read image");
        // Warmup vision download/load
        let _ = gate.decide(Input::ImageBytes(&bytes), &["warmup", "other"]).unwrap();

        let res = gate
            .decide(
                Input::ImageBytes(&bytes),
                &[
                    "a Windows desktop taskbar with app icons and clock",
                    "a photo of a mountain landscape",
                    "a handwritten math equation",
                    "a cat sitting on a sofa",
                ],
            )
            .unwrap();
        println!("image ({} bytes): {:?}", bytes.len(), res);
        return;
    }

    // Warmup downloads/loads weights once before timing
    let _ = gate.decide(Input::Text("warmup"), &["a", "b"]).unwrap();

    let t1 = gate
        .decide(
            Input::Text("payment webhook returned 502 bad gateway on checkout"),
            &["production incident", "billing inquiry", "feature request"],
        )
        .unwrap();
    println!(
        "text uncached: choice={:?} conf={:.3} latency={}ms",
        t1.choice, t1.confidence, t1.latency_ms
    );

    let t2 = gate
        .decide(
            Input::Text("payment webhook returned 502 bad gateway on checkout"),
            &["production incident", "billing inquiry", "feature request"],
        )
        .unwrap();
    println!(
        "text cached:   choice={:?} cached={} latency={}ms",
        t2.choice, t2.cached, t2.latency_ms
    );

    let red = image::RgbImage::from_pixel(224, 224, image::Rgb([230, 20, 20]));
    let mut png1 = Cursor::new(Vec::new());
    red.write_to(&mut png1, image::ImageFormat::Png).unwrap();
    let _ = gate
        .decide(Input::ImageBytes(png1.get_ref()), &["red", "blue"])
        .unwrap();

    let blue = image::RgbImage::from_pixel(224, 224, image::Rgb([15, 70, 235]));
    let mut png2 = Cursor::new(Vec::new());
    blue.write_to(&mut png2, image::ImageFormat::Png).unwrap();

    let v1 = gate
        .decide(
            Input::ImageBytes(png2.get_ref()),
            &["a bright blue color", "a bright red color", "a green forest"],
        )
        .unwrap();
    println!(
        "vision uncached: choice={:?} conf={:.3} latency={}ms",
        v1.choice, v1.confidence, v1.latency_ms
    );

    let v2 = gate
        .decide(
            Input::ImageBytes(png2.get_ref()),
            &["a bright blue color", "a bright red color", "a green forest"],
        )
        .unwrap();
    println!(
        "vision cached:   choice={:?} cached={} latency={}ms",
        v2.choice, v2.cached, v2.latency_ms
    );
}
