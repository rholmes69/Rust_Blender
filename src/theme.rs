//! Blender-inspired dark theme with a modern accent for the AI layer.

use egui::{
    Color32, CornerRadius, FontFamily, FontId, Stroke, TextStyle, Visuals, style::WidgetVisuals,
};

pub const BG_EDITOR: Color32 = Color32::from_rgb(48, 48, 48);
pub const BG_PANEL: Color32 = Color32::from_rgb(40, 40, 40);
pub const BG_HEADER: Color32 = Color32::from_rgb(35, 35, 35);
pub const BG_DEEP: Color32 = Color32::from_rgb(29, 29, 29);
pub const HEADER_PILL: Color32 = Color32::from_rgba_premultiplied(30, 30, 30, 215);
pub const WIDGET: Color32 = Color32::from_rgb(84, 84, 84);
pub const WIDGET_HOVER: Color32 = Color32::from_rgb(100, 100, 100);
pub const ACCENT: Color32 = Color32::from_rgb(71, 114, 179);
pub const ACCENT_BRIGHT: Color32 = Color32::from_rgb(92, 145, 222);
pub const AI_ACCENT: Color32 = Color32::from_rgb(158, 110, 255);
pub const AI_ACCENT2: Color32 = Color32::from_rgb(64, 196, 255);
pub const SELECT_OUTLINE: Color32 = Color32::from_rgb(241, 88, 0);
pub const ACTIVE_OUTLINE: Color32 = Color32::from_rgb(255, 170, 64);
pub const TEXT: Color32 = Color32::from_rgb(220, 220, 220);
pub const TEXT_DIM: Color32 = Color32::from_rgb(150, 150, 150);
pub const OK: Color32 = Color32::from_rgb(110, 205, 120);
pub const WARN: Color32 = Color32::from_rgb(240, 180, 70);
pub const ERR: Color32 = Color32::from_rgb(235, 90, 90);
pub const KEYFRAME: Color32 = Color32::from_rgb(232, 200, 60);
pub const REC: Color32 = Color32::from_rgb(235, 64, 72);

pub fn apply(ctx: &egui::Context) {
    let mut v = Visuals::dark();
    v.panel_fill = BG_PANEL;
    v.window_fill = Color32::from_rgb(38, 38, 38);
    v.extreme_bg_color = BG_DEEP;
    v.faint_bg_color = Color32::from_rgb(44, 44, 44);
    v.code_bg_color = BG_DEEP;
    v.window_stroke = Stroke::new(1.0, Color32::from_rgb(25, 25, 25));
    v.window_corner_radius = CornerRadius::same(8);
    v.menu_corner_radius = CornerRadius::same(6);
    v.selection.bg_fill = ACCENT;
    v.selection.stroke = Stroke::new(1.0, Color32::WHITE);
    v.hyperlink_color = ACCENT_BRIGHT;
    v.override_text_color = Some(TEXT);
    let r = CornerRadius::same(4);
    let w = |bg: Color32, stroke: Color32| WidgetVisuals {
        bg_fill: bg,
        weak_bg_fill: bg,
        bg_stroke: Stroke::new(1.0, stroke),
        corner_radius: r,
        fg_stroke: Stroke::new(1.0, TEXT),
        expansion: 0.0,
    };
    v.widgets.noninteractive = WidgetVisuals {
        bg_stroke: Stroke::new(1.0, Color32::from_rgb(30, 30, 30)),
        ..w(BG_PANEL, BG_PANEL)
    };
    v.widgets.inactive = w(Color32::from_rgb(64, 64, 64), Color32::from_rgb(44, 44, 44));
    v.widgets.hovered = w(WIDGET_HOVER, Color32::from_rgb(110, 110, 110));
    v.widgets.active = w(ACCENT, ACCENT_BRIGHT);
    v.widgets.open = w(Color32::from_rgb(70, 70, 70), Color32::from_rgb(90, 90, 90));
    ctx.set_visuals(v);
    ctx.all_styles_mut(|s| {
        s.spacing.item_spacing = egui::vec2(6.0, 5.0);
        s.spacing.button_padding = egui::vec2(7.0, 3.0);
        s.spacing.interact_size.y = 22.0;
        s.spacing.slider_width = 140.0;
        s.text_styles
            .insert(TextStyle::Body, FontId::new(13.5, FontFamily::Proportional));
        s.text_styles.insert(
            TextStyle::Button,
            FontId::new(13.5, FontFamily::Proportional),
        );
        s.text_styles.insert(
            TextStyle::Small,
            FontId::new(11.0, FontFamily::Proportional),
        );
        s.text_styles.insert(
            TextStyle::Heading,
            FontId::new(17.0, FontFamily::Proportional),
        );
        s.text_styles.insert(
            TextStyle::Monospace,
            FontId::new(12.0, FontFamily::Monospace),
        );
    });
}

pub fn color32(c: [f32; 3]) -> Color32 {
    Color32::from_rgb(
        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
    )
}
